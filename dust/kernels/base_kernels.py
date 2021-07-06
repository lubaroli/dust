from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseKernel(ABC):
    def __init__(
        self, analytic_grad=True,
    ):

        self.analytic_grad = analytic_grad

    @abstractmethod
    # def eval(self, X, Y, **kwargs):
    def eval(self, X, Y):
        """
        Evaluate kernel function and corresponding gradient terms for batch of inputs.

        Parameters
        ----------
        X : Tensor
            Data, of shape [batch, dim]
        Y : Tensor
            Data, of shape [batch, dim]
        kwargs : dict
            Kernel-specific parameters

        Returns
        -------
        K: Tensor
            Kernel Gram matrix, of shape [batch, batch].
        d_K_Xi: Tensor
            Kernel gradients wrt. first input X. Shape: [batch, batch, dim]
        """
        pass


class RBF(BaseKernel):
    """
        k(x, x') = exp( - || x - x'||**2 / (2 * ell**2))
    """

    def __init__(
        self, bandwidth=-1, bw_scale=1.0, analytic_grad=True, minimum_bw=1e-5, **kwargs
    ):
        super().__init__(analytic_grad,)
        self.ell = bandwidth
        self.ell_scale = bw_scale
        self.analytic_grad = analytic_grad
        self.minimum_bw = minimum_bw

    def compute_bandwidth(self, X, Y):
        """
            Older version.
        """

        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        pairwise_dists_sq = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        if self.ell < 0:  # use median trick
            try:
                h = torch.median(pairwise_dists_sq).detach()
            except Exception as e:
                print(pairwise_dists_sq)
                print(e)
        else:
            h = self.ell ** 2

        # Liu et al. 2017
        # h = h / np.log(X.shape[0])

        h = h / np.log(X.shape[0] + 1)

        # User-defined scaling
        h_scale = self.ell_scale
        h = h_scale * h

        # Clamp bandwidth
        tol = self.minimum_bw
        if isinstance(h, torch.Tensor):
            h = torch.clamp(h, min=tol)
        else:
            h = np.clip(h, a_min=tol, a_max=None)

        return h, pairwise_dists_sq

    def eval(
        self, X, Y,
    ):

        assert X.shape == Y.shape

        if self.analytic_grad:
            h, pw_dists_sq = self.compute_bandwidth(X, Y)
            K = (-pw_dists_sq / h).exp()
            d_K_Xi = K.unsqueeze(2) * (X.unsqueeze(1) - Y) * 2 / h
        else:
            # TODO: Implement autograd version
            raise NotImplementedError

        return (
            K,
            d_K_Xi,
        )
