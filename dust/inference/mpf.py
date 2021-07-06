import torch
import torch.distributions as dist
from KDEpy import bw_selection
from tqdm import trange

from .likelihoods import GaussianLikelihood
from .svgd import SVGD, bw_silverman

torch.autograd.set_detect_anomaly(True)


class MPF(SVGD):
    def __init__(
        self, init_particles, likelihood: GaussianLikelihood, bw=None, **kwargs
    ):
        super().__init__(**kwargs)
        assert (
            init_particles.ndim == 2
        ), "Particles must be two dimension with batch on dim 0."
        self.x = init_particles
        # initializes the prior based on init_particles
        self.update_prior(bw)
        self.likelihood = likelihood
        self.optimizer = self.optimizer_class(params=[self.x], **self.opt_args)

    def update_prior(self, bw):
        n_particles, dim_particles = self.x.shape
        if bw is None:
            bw = bw_silverman(self.x.flatten(1, -1), self.bw_scale)
        mix = dist.Categorical(torch.ones(n_particles))
        # TODO: check the best value for variance based on the bandwidth
        comp = dist.Independent(
            dist.MultivariateNormal(
                loc=self.x, covariance_matrix=bw ** 2 * torch.eye(dim_particles),
            ),
            reinterpreted_batch_ndims=0,
        )
        self.prior = dist.MixtureSameFamily(mix, comp)

    def phi(self, bw):
        """
            Uses manually-derived likelihood gradient.
        """
        x = self.x.detach().clone().requires_grad_(True)
        grad_prior = torch.autograd.grad(self.prior.log_prob(x).sum(), x)[0]
        obs = self.likelihood.sample(x)
        log_l = self.likelihood.log_prob(obs)
        # Analytic gradient has 2 parts, d_log Normal and d_log Model
        # grad_lik = (obs - x) / (self.likelihood.sigma ** 2)
        grad_lik = torch.autograd.grad(log_l.sum(), x)[0]
        score_func = grad_lik + grad_prior

        k_xx = self.kernel(x.flatten(1, -1), x.detach().clone().flatten(1, -1), bw=bw)
        grad_k = torch.autograd.grad(k_xx.sum(), x)[0]

        phi = grad_k + torch.tensordot(k_xx.detach(), score_func, dims=1) / x.size(0)
        return phi

    def step(self, bw):
        self.optimizer.zero_grad()
        self.x.grad = -self.phi(bw)
        self.optimizer.step()

    def optimize(
        self, action, new_obs, bw=None, n_steps=100, debug=False,
    ):
        if new_obs is not None:
            self.likelihood.condition(action, new_obs)
        if bw is None:
            # bw = bw_silverman(self.x.flatten(1, -1))
            # bw = bw_selection.improved_sheather_jones(self.x.view(-1, 1).numpy())
            bw = bw_selection.silvermans_rule(self.x.view(-1, 1).numpy())
            bw = bw * self.bw_scale
        if debug:
            iterator = trange(n_steps, position=0, leave=True)
        else:
            iterator = range(n_steps)
        grads = []
        for _ in iterator:
            self.step(bw)
            if debug:
                iterator.set_postfix(loss=self.x.grad.detach().norm(), refresh=False)
            grads.append(self.x.grad.detach().norm())

        self.update_prior(bw)
        return torch.as_tensor(grads), bw
