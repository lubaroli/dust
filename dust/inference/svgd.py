import torch
import torch.distributions as dist
import torch.optim as optim
from scipy.stats import scoreatpercentile as p_score
from tqdm import trange

torch.autograd.set_detect_anomaly(True)


def _select_sigma(x: torch.Tensor, percentile: int = 25):
    """
    Returns the smaller of std or normalized IQR of x over axis 0. Code originally from:
    https://github.com/statsmodels/statsmodels/blob/master/statsmodels/nonparametric/bandwidths.py
    References
    ----------
    Silverman (1986) p.47
    """
    # normalize = norm.ppf(.75) - norm.ppf(.25)
    normalize = 1.349
    IQR = (p_score(x, 100 - percentile) - p_score(x, percentile)) / normalize
    std_dev = torch.std(x, axis=0)
    if IQR > 0 and IQR < std_dev.min():
        return IQR
    else:
        return std_dev


def squared_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes squared distance matrix between two arrays of row vectors.

    Code originally from:
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm)
    return res.clamp(min=0)  # avoid negative distances due to num precision


def bw_median(
    x: torch.Tensor, y: torch.Tensor = None, bw_scale: float = 1.0, tol: float = 1.0e-5
) -> torch.Tensor:
    if y is None:
        y = x.detach().clone()
    pairwise_dists = squared_distance(x, y).detach()
    h = torch.median(pairwise_dists)
    # TODO: double check which is correct
    # h = torch.sqrt(0.5 * h / torch.tensor(x.shape[0] + 1)).log()
    h = torch.sqrt(0.5 * h) / torch.tensor(x.shape[0] + 1.0).log()
    return bw_scale * h.clamp_min_(tol)


def bw_silverman(x: torch.Tensor, bw_scale: float = 1.0) -> torch.Tensor:
    """
    Silverman's Rule of Thumb. Code originally from:
    https://github.com/statsmodels/statsmodels/blob/master/statsmodels/nonparametric/bandwidths.py

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Unused
    Returns
    -------
    bw : float
        The estimate of the bandwidth
    Notes
    -----
    Returns .9 * A * n ** (-1/5.) where ::
       A = min(std(x), IQR/1.349)
       IQR = score_at_percentile(x, [75,25]))
    References
    ----------
    Silverman, B.W. (1986) `Density Estimation.`
    """
    A = _select_sigma(x)
    n = len(x)
    return bw_scale * (0.9 * A * n ** (-0.2))


def get_gmm(
    x: torch.Tensor, weights: torch.Tensor, covariance: torch.Tensor
) -> dist.mixture_same_family.MixtureSameFamily:
    mix = dist.Categorical(weights)
    comp = dist.Independent(dist.MultivariateNormal(x.detach(), covariance), 1)
    return dist.mixture_same_family.MixtureSameFamily(mix, comp)


def default_kernel(x: torch.Tensor, y: torch.Tensor = None, bw=0.69):
    if y is None:
        # Flatten input as Kernel is 1-D
        y = x.detach().clone().flatten(1, -1)
    pairwise_dists = squared_distance(x, y)
    # compute the rbf kernel
    Kxy = torch.exp(-pairwise_dists / bw ** 2 / 2)
    return Kxy


class SVGD:
    """
    An implementation of Stein variational gradient descent.

    Adapted from: https://github.com/activatedgeek/svgd
    """

    def __init__(
        self,
        kernel=None,
        bw_scale=1.0,
        n_particles=None,
        n_steps=100,
        optimizer_class=optim.Adam,
        **opt_args
    ):
        if kernel is None:
            kernel = default_kernel
        self.kernel = kernel
        self.bw_scale = bw_scale
        self.optimizer_class = optimizer_class
        self.opt_args = opt_args
        self.n_steps = n_steps
        self.n_particles = n_particles

    def phi(self, x, log_p, h):
        score_func = torch.autograd.grad(log_p(x).sum(), x)[0]

        k_xx = self.kernel(x, x.detach(), h)
        grad_k = -torch.autograd.grad(k_xx.sum(), x)[0]

        phi = (k_xx.detach().matmul(score_func) + grad_k) / x.size(0)

        return phi

    def step(self, x, optimizer, log_p, bw):
        optimizer.zero_grad()
        x.grad = -self.phi(x, log_p, bw)
        optimizer.step()

    def score_matrix(self, x, log_p):
        x = x.detach().clone().requires_grad_()
        s_matrix = torch.autograd.grad(log_p(x).sum(), x)[0]
        return s_matrix

    def discrepancy(self, x, log_p):
        s_matrix = self.score_matrix(x, log_p)
        bw = bw_median(x, x)
        k_matrix = default_kernel(x, x, bw)
        sst = s_matrix @ s_matrix.t()
        d = x.shape[1]
        return (k_matrix * (sst + d / bw ** 2)).detach().mean().sqrt()

    def optimize(
        self,
        log_p,
        initial_particles: torch.Tensor = None,
        prior=None,
        debug=False,
        bw=0.69,
    ):
        if initial_particles is not None:
            x = initial_particles.detach().clone().requires_grad_(True)
        elif prior is not None:
            x = prior.sample(torch.Size([self.n_particles])).requires_grad_(True)
        else:
            raise RuntimeError(
                "Either initial_particles or prior must be specified for SVGD"
            )

        optimizer = self.optimizer_class(params=[x], **self.opt_args)

        if self.kernel is default_kernel:
            bw = bw_median(x, x)  # using median trick

        if debug:
            iterator = trange(self.n_steps, position=0, leave=True)
        else:
            iterator = range(self.n_steps)

        for _ in iterator:
            self.step(x, optimizer, log_p, bw)
            if debug:
                iterator.set_postfix(loss=x.grad.detach().norm(), refresh=False)

        return x.detach()
