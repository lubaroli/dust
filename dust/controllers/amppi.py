import torch
from .base import BaseController
from ..utils.utf import MerweScaledUTF


class AMPPI(BaseController):
    """Implements an variation of the IT-MPC controller as defined in [1]_ for
    OpenAI Gym environments.

    .. [1] Williams et al., 2017 'Information Theoretic MPC for Model-Based
        Reinforcement Learning'
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hz_len,
        n_samples,
        lambda_=1.0,
        a_cov=None,
        inst_cost_fn=None,
        term_cost_fn=None,
        params_sampling="extended",
        **kwargs
    ):
        """Constructor for AMPPI.

        :param observation_space: A Box object defining the action space.
        :type observation_space: gym.spaces.Space
        :param action_space: A Box object defining the action space.
        :type action_space: gym.spaces.Space
        :param hz_len: Number of time steps in the control horizon.
        :type hz_len: int
        :param lambda_: Controller regularization parameter. Defaults to 1.0.
        :param a_cov: covariance matrix of the actions multiplicative Gaussian
            noise. Effectively determines the amount of exploration
            of the system. If None, an appropriate identity matrix is used.
            Defaults to None.
        :type a_cov: torch.Tensor
        :key inst_cost_fn: A function that receives a trajectory and returns
            the instantaneous cost. Must be defined if no `term_cost_fn` is
            given. Defaults to None.
        :type kwargs: function
        :key term_cost_fn: A function that receives a state
            and returns its terminal cost. Must be defined if no
            `inst_cost_fn` is given. Defaults to None.
        :param params_sampling: Can be set to either 'none, 'single',
            'extended', or a Transformer object. If 'none', mean values of the
            parameter distribution are used if available. Otherwise, default
            model parameters are used. If 'single', one sample per rollout is
            taken and used for all `n_samples` trajectories. If 'extended',
            `n_samples` are taken per rollout, meaning each trajectory has their
            own sampled parameters. Finally, if a Transformer is provided it
            will be used *instead* of sampling parameters. Defaults to
            'extended'.
        :type params_sampling: str or utils.utf.MerweScaledUTF
        :key init_actions:  A tensor of dimension `hz_len` by
            `action_space.shape` containing the initial set of control actions.
            If None, the sequence is initialized to zeros. Defaults to None.
        :type kwargs: torch.Tensor

        .. note::
            * Actions will be clipped according bounding action space regardless
              of the covariance set. Effectively `epsilon <= (max_a - min_a)`.

            * Sampling is ignored when a transformer object is provided to
              method `update_actions`.
        """
        super(AMPPI, self).__init__(
            observation_space,
            action_space,
            hz_len,
            inst_cost_fn,
            term_cost_fn,
            **kwargs
        )
        self.n_samples = n_samples
        self.lambda_ = lambda_
        if a_cov is None:
            a_cov = torch.eye(self.dim_a)
        a_loc = torch.zeros(self.dim_a)
        self.a_dist = torch.distributions.multivariate_normal.MultivariateNormal(
            a_loc, a_cov
        )
        self.a_pre = torch.inverse(a_cov)

        if not params_sampling or params_sampling == "none":
            self.__params_sample_shape = None
            self.__tf = None
        elif params_sampling == "single":
            self.__params_sample_shape = 1
            self.__tf = None
        elif params_sampling == "extended":
            self.__params_sample_shape = n_samples
            self.__tf = None
        elif isinstance(params_sampling, MerweScaledUTF):
            self.__params_sample_shape = None
            self.__tf = params_sampling
        else:
            raise ValueError(
                "Invalid value for 'params_sampling': {}".format(params_sampling)
            )
        self.__params_sampling = params_sampling

    @property
    def params_sampling(self):
        return self.__params_sampling

    def _sample_trajectories(self, model, state, actions):
        """Sample trajectories.

        :param model: A model object which provides a `step` function to
            generate the system rollouts. If params_sampling is used, it must
            also implement a `sample_params` function for the parameters of the
            transition function.
        :type model: models.base.BaseModel
        :param state: The initial state of the system.
        :type state: torch.Tensor
        :returns: A tuple of (actions, states, eps) for `n_samples` rollouts.
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        if actions is None:
            # eps shape is `n_samples` x `hz_len` x `dim_a`
            eps = self.a_dist.sample(sample_shape=[self.n_samples, self.hz_len])
            acts = torch.add(eps, self.a_seq)
        else:
            acts = actions
            eps = torch.add(acts, -self.a_seq)
        states = torch.zeros(self.n_samples, self.hz_len + 1, self.dim_s)
        states[:, 0] = state.repeat(self.n_samples, 1)
        # shape of sampled_params is a dict with `uncertain_params` as keys and
        # tensors of size ``[(1 | `n_samples`), 1]`` samples
        if self.__params_sample_shape:
            sampled_params = model.sample_params(self.__params_sample_shape)
        else:
            sampled_params = None
        for t in range(self.hz_len):
            states[:, t + 1] = model.step(states[:, t], acts[:, t], sampled_params)
        return acts, states, eps

    def _sample_sigma_trajectories(self, model, state, actions):
        """Sample trajectories using Unscented Transform.

        :param model: A model object which provides a `step` function to
            generate the system rollouts.
        :type model: models.base.BaseModel
        :param state: The initial state of the system.
        :type state: torch.Tensor
        :returns: A tuple of (actions, states, eps) for `n_samples` * `tf.pts`
            rollouts.
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        if actions is None:
            # eps shape is `n_samples` x `hz_len` x `dim_a`.
            eps = self.a_dist.sample(sample_shape=[self.n_samples, self.hz_len])
            acts = torch.add(eps, self.a_seq)
        else:
            acts = actions
            eps = torch.add(acts, -self.a_seq)

        # Create a matrix to store all sigma points for each trajectory and for
        # each step, last dim is the number of sigma points
        try:
            covariance = model.params_dist.covariance_matrix
            mean = model.params_dist.mean
        except AttributeError:
            try:
                covariance = model.params_dist.variance.diag()
                mean = model.params_dist.mean
            except AttributeError:
                idx = model.params_dist.a.argmax()
                covariance = model.params_dist.xs[idx].S
                mean = model.params_dist.xs[idx].m
        params_sp = self.__tf.compute_sigma_points(mean, covariance)

        # Create tensors to be used on forward model. When using sigma points,
        # the number of trajectory is effectively `n_samples` * `tf.pts`. We'll
        # repeat `tf.pts` times each row of `actions` and `n_samples` times the
        # `params_sp` tensor, so every action is applied to each sigma point.
        acts_sp = acts.repeat(1, self.__tf.pts, 1).view(-1, self.hz_len, self.dim_a)
        sampled_params = model.to_params_dict(params_sp.T.repeat(self.n_samples, 1))
        states_sp = torch.zeros(
            self.n_samples * self.__tf.pts, self.hz_len + 1, self.dim_s
        )
        states_sp[:, 0] = state.expand_as(states_sp[:, 0])
        for t in range(self.hz_len):
            states_sp[:, t + 1] = model.step(
                states_sp[:, t], acts_sp[:, t], sampled_params
            )
        return acts, states_sp, eps

    def _compute_cost(self, states, eps):
        """Estimate trajectories cost.

        :param states: A tensor with the states of each trajectory.
        :type states: torch.Tensor
        :param eps: A tensor with the difference of the current planned action
            sequence and the actions on each trajectory.
        :type eps: torch.Tensor
        :returns: A tensor with the costs for the given trajectories.
        :rtype: torch.Tensor
        """
        # Need to use reshape instead of view because slice is not contiguous
        inst_costs = self.inst_cost_fn(states[:, 1:].reshape(-1, self.dim_s))
        inst_costs = inst_costs.view(-1, self.hz_len).sum(dim=1)
        term_costs = self.term_cost_fn(states[:, -1])
        if self.__tf:
            # Weight trajectories using UTF sigma weights
            inst_costs = torch.matmul(
                inst_costs.view(-1, self.__tf.pts), self.__tf.loc_weights
            )
            term_costs = torch.matmul(
                term_costs.view(-1, self.__tf.pts), self.__tf.loc_weights
            )

        # To compute ctrl_costs in a single batch for all hz_len and all
        # n_samples, first we compute the hz_len x dim_a cost matrix, result
        # size is n_samples x hz_len x hz_len. Then take the trace of the
        # hz_len x hz_len matrices, result is dim n_samples.
        ctrl_costs = self.lambda_ * (
            torch.matmul(torch.matmul(self.a_seq, self.a_pre), eps.transpose(1, 2))
        ).diagonal(dim1=1, dim2=2).sum(dim=1)
        costs = term_costs + inst_costs + ctrl_costs  # shape is n_samples
        return costs

    def update_actions(self, model, state, actions=None):
        """Computes the next control action and the incurred cost. Updates the
        controller next control actions.

        :param model: A model with a `step(states, actions, params)` function
            to compute the next state for a set of trajectories.
        :type model: models.base.BaseModel
        :param state: A with the system initial state.
        :type state: torch.Tensor
        :param actions: Actions sampled outside the controller scope. Must have
            shape `n_samples` x `hz_len` x `dim_a`. If `None`, actions are
            sampled by the controller.
        :type actions: torch.Tensor
        :returns: A tuple of `(cost, omega)` with the expected cost of the new
             actions and weights of the computed trajectories.
        :rtype: (float, torch.Tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float)
        if self.__tf:
            acts, states, eps = self._sample_sigma_trajectories(model, state, actions)
        else:
            acts, states, eps = self._sample_trajectories(model, state, actions)
        costs = self._compute_cost(states, eps)  # raw costs
        beta = costs.min()
        log_costs = (-1 / self.lambda_) * (costs - beta)
        # log normalising constant
        eta = log_costs.logsumexp(0)
        # log costs, tensor of size n_samples
        omega = log_costs - eta
        self.a_seq += torch.tensordot(omega.exp(), eps, dims=1)
        # Even though tau has been clipped, epsilon can be of mag
        # (u_max-u_min) so we need to clip U again
        self.a_seq = torch.clamp(self.a_seq, self.min_a, self.max_a)
        return costs, states, acts, omega
