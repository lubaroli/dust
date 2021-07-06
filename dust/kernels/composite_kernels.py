from abc import ABC, abstractmethod

import torch

from .base_kernels import RBF


class CompositeKernel(ABC):
    def __init__(
        self, base_kernel=RBF(), ctrl_dim=1, indep_controls=True, **kwargs,
    ):

        self.ctrl_dim = ctrl_dim
        self.base_kernel = base_kernel
        self.indep_controls = indep_controls

    @abstractmethod
    def eval(self, X, Y, **kwargs):
        """
        Parameters
        ----------
        X : tensor of shape [batch, dim]
        Y : tensor of shape [batch, dim]
        kwargs :

        Returns
        -------

        """
        pass


class iid_mp(CompositeKernel):
    def eval(self, X, Y, **kwargs):
        X = X.view(X.shape[0], -1, self.ctrl_dim)
        Y = Y.view(Y.shape[0], -1, self.ctrl_dim)

        # m: batch, h: horizon, d: ctrl_dim
        m, h, d = X.shape

        # Keep another batch-dim for grad. mult. later on.
        kernel_Xj_Xi = torch.zeros(m, m, h, d)

        # shape : (m, h, d)
        d_kernel_Xi = torch.zeros(m, m, h, d)

        if self.indep_controls:
            for i in range(h):
                for q in range(self.ctrl_dim):
                    k_tmp, dk_tmp = self.base_kernel.eval(
                        X[:, i, q].reshape(-1, 1), Y[:, i, q].reshape(-1, 1),
                    )
                    kernel_Xj_Xi[:, :, i, q] += k_tmp
                    d_kernel_Xi[:, :, i, q] += dk_tmp.squeeze(2)
        else:
            for i in range(h):
                k_tmp, dk_tmp = self.base_kernel.eval(X[:, i, :], Y[:, i, :],)
                kernel_Xj_Xi[:, :, i, :] += k_tmp.unsqueeze(2)
                d_kernel_Xi[:, :, i, :] += dk_tmp

        kernel_Xj_Xi = kernel_Xj_Xi.reshape(m, m, h * d)
        d_kernel_Xi = d_kernel_Xi.reshape(m, m, h * d)

        return kernel_Xj_Xi, d_kernel_Xi
