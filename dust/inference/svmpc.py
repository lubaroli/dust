import torch
from gpytorch.kernels import RBFKernel
from KDEpy import bw_selection
from tqdm import trange

from ..kernels.base_kernels import RBF
from ..kernels.composite_kernels import iid_mp
from .likelihoods import CostLikelihood
from .svgd import SVGD, bw_median, get_gmm

torch.autograd.set_detect_anomaly(True)


class SVMPC(SVGD):
    def __init__(
        self,
        init_particles: torch.Tensor,
        prior: torch.distributions.Distribution,
        likelihood: CostLikelihood,
        roll_strategy="repeat",
        weighted_prior: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.theta = init_particles
        self.prior = prior
        self.likelihood = likelihood
        self.w_prior = weighted_prior
        self.roll_strategy = roll_strategy
        self.optimizer = self.optimizer_class(params=[self.theta], **self.opt_args)

    def phi(self, log_p, bw, sigma):
        """
            Uses manually-derived likelihood gradient.
        """
        # costs: shape [num_ctrl_samples, num_part]
        # actions: shape [num_ctrl_samples, num_part, horizon, ctrl_dim]
        x = self.theta.detach().clone().requires_grad_(True)

        # Prior gradient
        grad_pri = torch.autograd.grad(self.prior.log_prob(x).sum(), x)[0]

        # Likelihood gradient...
        log_l, costs, actions = log_p(x)

        # ... analytic
        n_samples = actions.shape[0]
        alpha = self.likelihood.alpha
        cost_weights = torch.zeros(n_samples, self.n_particles)
        for i in range(self.n_particles):
            cost_weights[:, i] = torch.softmax(-costs[:, i] * alpha, dim=0)
        d_log_pi = (actions - x) / sigma ** 2
        cost_weights = cost_weights.unsqueeze(-1).unsqueeze(-1)
        grad_lik = (cost_weights * d_log_pi).sum(0)  # sum across samples

        score_func = grad_lik + grad_pri

        # ... or autodiff
        # grad_lik = torch.autograd.grad(log_l.sum(), x)[0]
        # score_func = grad_lik + grad_pri

        # Kernel gradient
        phi, grad_k = 0, 0
        if isinstance(self.kernel, (RBF, iid_mp)):
            # Gradient for own kernels
            x = x.reshape(self.n_particles, -1)
            self.kernel.ell = bw
            k_XX, grad_k = self.kernel.eval(x, x.detach().clone(),)
            # Message passing kernel
            score_func = score_func.view(1, self.n_particles, -1)
            grad = (k_XX.detach() * score_func).mean(1)
            rep = grad_k.mean(1)
            phi = grad + rep
            phi = phi.reshape(self.theta.shape)

        elif isinstance(self.kernel, (RBFKernel)):
            # Gradient for GPyTorch kernels
            self.kernel.lenghtscale = bw
            k_xx = self.kernel(
                x.flatten(1, -1), x.detach().clone().flatten(1, -1)
            ).evaluate()
            grad_k = torch.autograd.grad(k_xx.sum(), x)[0]
            phi = grad_k + torch.tensordot(k_xx, score_func, 1) / x.size(0)

        return phi

    def step(self, state, params_dist, bw, sigma):
        def log_p(theta):
            costs, actions = self.likelihood.sample(theta, state, params_dist)
            return self.likelihood.log_prob(costs), costs, actions

        self.optimizer.zero_grad()
        self.theta.grad = -self.phi(log_p, bw, sigma)
        self.optimizer.step()
        self.theta.detach()

    def optimize(
        self, state, params_dist, bw=None, n_steps=None, debug=False,
    ):
        if bw is None:
            # using median trick
            # bw = bw_median(
            #     self.theta.flatten(1, -1), self.theta.flatten(1, -1), self.bw_scale
            # )
            bw = bw_selection.silvermans_rule(self.theta.view(-1, 1).numpy())

        try:
            sigma = self.likelihood.controller.a_dist.covariance_matrix
        except AttributeError:
            sigma = torch.inverse(self.likelihood.controller.a_pre)
        sigma = sigma.diag().sqrt()

        if n_steps is None:
            n_steps = self.n_steps

        if debug:
            iterator = trange(n_steps, position=0, leave=True)
        else:
            iterator = range(n_steps)

        for _ in iterator:
            self.step(state, params_dist, bw, sigma)
            if debug:
                iterator.set_postfix(
                    loss=self.theta.grad.detach().norm(), refresh=False
                )

    def get_weights(self, state, params_dist, fast_pred=True):
        """
        Reestimates particles weights after gradient step. Will recompute
        roll-outs using the likelihood.
        """
        if fast_pred:
            costs = self.likelihood.last_costs
        else:
            costs, _ = self.likelihood.sample(self.theta, state, params_dist)
        log_l = self.likelihood.log_prob(costs)
        log_p = self.prior.log_prob(self.theta)
        log_w = log_l + log_p
        return (log_w - log_w.logsumexp(0)).exp()

    def roll(self, steps=-1, strategy="repeat"):
        # roll along time axis
        self.theta = self.theta.roll(steps, dims=-2)
        if strategy == "repeat":
            # repeat last action
            self.theta[..., -1, :] = self.theta[..., -2, :]
        elif strategy == "resample":
            # sample a new action from prior for last step
            self.theta[..., -1, :] = self.prior.sample([self.n_particles])[..., -1, :]
        elif strategy == "mean":
            # last action will be the mean of each policy
            self.theta[..., -1, :] = self.theta.mean(dim=-2)
        else:
            raise ValueError("{} is an invalid roll strategy.".format(strategy))

        # update optimizer params
        self.optimizer.param_groups[0]["params"][0] = self.theta

    def update_prior(self, weights=None):
        weights = torch.ones(self.theta.shape[0]) if weights is None else weights
        if self.w_prior is False:
            mix = torch.ones_like(weights)
        else:
            mix = weights
        self.prior = get_gmm(
            self.theta,
            mix,
            self.prior.component_distribution.base_dist.covariance_matrix,
        )

    def forward(self, state, params_dist, steps=-1, fast_pred=True):
        """
            Called after the SVGD loop.
            1. Evaluate weights on updated stein particles : requires
            re-estimating the gradients on likelihood and prior for the new
             parameter values. Likelihood gradients therefore require additional
             sampling.
            2. Pick the best particle according to weights.
            3. Sets action sequence according to particle.
            4. Shift particles.
            5. Update prior using new weights.
        """

        # to compute the weights, we may either re-sample the likelihood to get the
        # expected cost of the new θ_i or set `fast_pred` to re-use the costs computed
        # during the `optimize` step to save computation.
        with torch.no_grad():
            p_weights = self.get_weights(state, params_dist, fast_pred)

        # Pick best particle
        i_star = p_weights.argmax()
        a_seq = self.theta[i_star].detach().clone()

        # roll thetas for next step
        # TODO: Should this be clamped?
        self.roll(steps, self.roll_strategy)
        # and set the mixture of the new prior
        self.update_prior(p_weights)
        return a_seq, p_weights
