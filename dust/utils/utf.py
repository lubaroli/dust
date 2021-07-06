from __future__ import division

import torch


class MerweScaledUTF:
    """Creates a transformer object using Merwe Scaled Sigma Points [1] for the
    unscented transform. Based on the code available at [2].

    .. note:: Sigma points are parameterized using default alpha, beta, kappa
    terms, according to [2] this is currently the de-facto standard in most
    publications.

    .. [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilistic
       Inference in Dynamic State-Space Models" (Doctoral dissertation)
    .. [2] https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, n, alpha=1e-3, beta=2, kappa=0, sqrt_method=None):
        """Constructor for the MerweScaledUTF.

        :param n: Dimensionality of the state. 2n+1 weights will be generated.
        :type n: int
        :param alpha: Defines the spread of the sigma points around the mean.
            Usually a small positive value. Defaults to 1e-3.
        :type alpha: float
        :param beta: Incorporates prior knowledge of the distribution of the
            mean. For Gaussian distributions, beta=2 is optimal. Defaults to 2.
        :type beta: float
        :param kappa: Secondary scaling parameter usually set to 0 or to 3-n.
            Defaults to 0.
        :type kappa: float
        :param sqrt_method: Defines how we compute the square root of a matrix,
            which has no unique answer. Cholesky is the default choice due to
            its speed. As of [1] this was not a well researched area and
            different choices affect how the sigma points are arranged relative
            to the eigenvectors of the covariance matrix. If the default is not
            used, the alternative method must return an upper triangular matrix
            If None, defaults to `torch.linalg.cholesky`.
        :type sqrt_method: function
        """
        self.n = n
        self.pts = 2 * n + 1
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = (
                lambda A: torch.linalg.cholesky(A.transpose(-2, -1).conj())
                .transpose(-2, -1)
                .conj()
            )
        else:
            self.sqrt = sqrt_method
        self._compute_weights()

    def __repr__(self):
        args = (
            "n: {},\n alpha: {},\n beta: {},\n kappa: {},\n"
            "loc_weights:\n{},\ncov_weights:\n{},\n".format(
                self.n,
                self.alpha,
                self.beta,
                self.kappa,
                self.__loc_weights,
                self.__cov_weights,
            )
        )
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def loc_weights(self):
        """Tensor with weights for the mean computation."""
        return self.__loc_weights

    @property
    def cov_weights(self):
        """Tensor with weights for the covariance computation."""
        return self.__cov_weights

    def _compute_weights(self):
        """Computes the weights for the Merwe scaled sigma points."""
        n = self.n
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n
        c = 0.5 / (n + lambda_)
        self.__cov_weights = torch.ones(self.pts, dtype=torch.float) * c
        self.__loc_weights = torch.ones(self.pts, dtype=torch.float) * c
        self.__cov_weights[0] = lambda_ / (n + lambda_) + (
            1 - self.alpha ** 2 + self.beta
        )
        self.loc_weights[0] = lambda_ / (n + lambda_)

    def compute_sigma_points(self, mu, K):
        """Computes the sigma points for a given distribution given its mean
        (mu) and covariance (K) matrix.

        :param mu: A tensor-like object of length :attr:`n` containing the
            distribution means. Can be a scalar if 1D.
        :param K: A tensor-like object of size (:attr:`n`, :attr:`n`)
            containing the covariance matrix of the distribution. If argument
            is scalar, assume form as torch.eye(n)*P.

        :return tensor sigmas : Two dimensional tensor of of size [n, 2n+1]
            containing the sigma points. Each row contains all of the sigmas
            for one dimension in the state space.
        """
        # convert all floating points to torch.float64 for numerical precision
        mu = torch.as_tensor(mu, dtype=torch.float)
        K = torch.as_tensor(K, dtype=torch.float)
        if self.n != mu.size(0):
            raise ValueError(
                "expected size(x) {}, but size is {}".format(self.n, mu.size(0))
            )
        n = self.n
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n) * K)

        sigmas = torch.zeros(n, self.pts, dtype=torch.float)
        sigmas[:, 0] = mu
        # Use broadcasting to compute remaining points
        sigmas[:, 1 : n + 1] = U + mu.view(-1, 1)
        sigmas[:, n + 1 : self.pts] = -U + mu.view(-1, 1)
        return sigmas

    def unscented_transform(self, sigmas):
        """Computes unscented transform of a set of sigma points using
        transformer weights. Returns the mean and covariance in a tuple.

        :param sigmas: tensor of size [n, 2n+1], with n the input dimension,
            contains the transformed sigma points
        :returns mu: tensor of size [n] with mean vector
        :returns K: tensor of size [n, n] with the covariance matrix
        """
        # new mean is just the sum of the sigmas * weight
        mu = sigmas @ self.loc_weights  # =\Sigma_(i=0)^(2n) (W^m[i]*X[i])
        # get residuals using broadcasting
        y = sigmas - mu.view(-1, 1)
        # new covariance is the sum of the outer product of the residuals
        # times the weights
        K = y @ torch.diag(self.__cov_weights) @ y.t()
        return mu, K
