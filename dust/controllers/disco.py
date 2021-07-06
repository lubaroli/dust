import torch
from .base import BaseController
from ..utils.utf import MerweScaledUTF

Empty = torch.Size([])


class MultiDISCO(BaseController):
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
        n_policies,
        action_samples,
        temperature=1.0,
        ctrl_penalty=1.0,
        a_cov=None,
        inst_cost_fn=None,
        term_cost_fn=None,
        params_sampling=True,
        params_samples=4,
        params_log_space=False,
        init_actions=None,
        **kwargs
    ):
        """Constructor for MultiDISCO.

        :param observation_space: A Box object defining the action space.
        :type observation_space: gym.spaces.Space
        :param action_space: A Box object defining the action space.
        :type action_space: gym.spaces.Space
        :param hz_len: Number of time steps in the control horizon.
        :type hz_len: int
        :param temperature: Controller temperature parameter. Defaults to 1.0.
        :type temperature: float
        :param ctrl_penalty: Proportional cost penalty for controller actions.
            Ranges from 0, no control cost, to 1, standard smoothing. Defaults
            to 1.0.
        :type ctrl_penalty: float
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
            taken and used for all `n_actions` trajectories. If 'extended',
            `n_actions` samples are taken per rollout, meaning each trajectory
            has their own sampled parameters. Finally, if a Transformer is
            provided it will be used *instead* of sampling parameters. Defaults
            to 'extended'.
        :type params_sampling: str or utils.utf.MerweScaledUTF
        :param init_actions:  A tensor of dimension `n_pol` x `hz_len` x
            `action_space.shape` containing the initial set of control actions.
            If None, the sequence is initialized to zeros. Defaults to None.
        :type init_actions: torch.Tensor

        .. note::
            * Actions will be clipped according bounding action space regardless
              of the covariance set. Effectively `epsilon <= (max_a - min_a)`.
        """
        super().__init__(
            observation_space,
            action_space,
            hz_len,
            inst_cost_fn,
            term_cost_fn,
            **kwargs
        )
        self.n_pol = n_policies
        self.n_actions = action_samples
        self.temp = temperature
        self.a_reg = temperature * (1 - ctrl_penalty)
        if a_cov is None:
            a_cov = torch.eye(self.dim_a)
        a_loc = torch.zeros(self.dim_a)
        self.a_dist = torch.distributions.multivariate_normal.MultivariateNormal(
            a_loc, a_cov
        )
        self.a_pre = torch.inverse(a_cov)
        # Since we have multiple controllers, we now have a matrix of action
        # sequences.
        if init_actions is None:
            self.a_mat = torch.zeros(self.n_pol, *self.a_seq.shape)
        else:
            assert init_actions.shape == (
                self.n_pol,
                *self.a_seq.shape,
            ), "Initial actions shape mismatch."
            self.a_mat = init_actions.clone()
        self.a_mix = torch.ones(self.n_pol)

        self._params_sampling = params_sampling
        self._params_log_space = params_log_space
        if (
            params_sampling is False
            or params_sampling is None
            or params_sampling == "none"
        ):
            self.n_params = 1
            self._params_shape = None
            self._tf = None
        elif params_sampling is True:
            self.n_params = params_samples
            self._params_shape = [self.n_params]
            self._tf = None
        elif isinstance(params_sampling, MerweScaledUTF):
            assert (
                self._params_log_space is False
            ), "Distribution must not be on log space if using UTF."
            # TODO: check is this is the correct behaviour for _sigma_rollouts
            self.n_params = 1
            self._params_shape = None
            self._tf = params_sampling
        else:
            raise ValueError(
                "Invalid value for 'params_sampling': {}".format(params_sampling)
            )
        # total amount of rollouts
        self.n_rollouts = self.n_params * self.n_actions * self.n_pol

    def _rollout(self, state, model, params_dist, ext_actions):
        """Perform rollouts based on current state and control plan.

        :param model: A model object which provides a `step` function to
            generate the system rollouts. If params_sampling is used, it must
            also implement a `sample_params` function for the parameters of the
            transition function.
        :type model: models.base.BaseModel
        :param state: The initial state of the system.
        :type state: torch.Tensor
        :param ext_actions: A matrix of shape `n_actions` x `n_pol` x `hz_len` x
            `dim_a` action sequences.
        :type actions: torch.Tensor
        :returns: A tuple of (actions, states, eps) for `n_actions` rollouts.
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        if ext_actions is None:
            # sample actions based on `n_actions`
            eps = self.a_dist.sample(
                sample_shape=[self.n_actions, self.n_pol, self.hz_len]
            )
            actions = torch.add(eps, self.a_mat)
        else:
            # create eps from the difference between external actions and planned a_seq
            actions = ext_actions.clone()
            eps = torch.add(actions, -self.a_seq)

        # prepare tensors for batch rollout
        if self._params_shape is not None:  # sample params from `params_dist`
            event_shape = params_dist.event_shape
            dim_params = 1 if event_shape == Empty else event_shape[0]
            # `params` is a tensor of size `n_params` x `dim_params`
            params = params_dist.sample(self._params_shape)
            params_log_p = params_dist.log_prob(params)
            if self._params_log_space is True:
                params = params.exp()
            # repeat and reshape so each param is applied to all `n_actions`
            # (just using repeat would replicate the batch and wouldn't work)
            params = params.repeat(1, self.n_actions * self.n_pol).reshape(
                -1, dim_params
            )
            # create dict with `uncertain_params` as keys for `model.step()`
            params_dict = model.params_to_dict(params)
        else:  # use default `model` params
            params_dict = None
            params_log_p = None
        # flatten policies into single dim and repeat for each sampled param
        actions = actions.reshape(-1, self.hz_len, self.dim_a).repeat(
            self.n_params, 1, 1
        )
        # expand initial state to the total amount of rollouts
        states = state.expand(self.n_rollouts, 1, -1).clone()

        # generate rollout
        for t in range(self.hz_len):
            states = torch.cat(
                [
                    states,
                    model.step(states[:, t], actions[:, t], params_dict).unsqueeze(1),
                ],
                dim=1,
            )

        # restore vectors dims, `n_params` is now first dimension
        states = states.reshape(
            -1, self.n_actions, self.n_pol, self.hz_len + 1, self.dim_s
        )
        actions = actions.reshape(
            -1, self.n_actions, self.n_pol, self.hz_len, self.dim_a
        )
        return states, actions, eps, params_log_p

    def _sigma_rollout(self, state, model, params_dist, ext_actions):
        """Perform rollouts based on current state and control actions using
        Unscented Transform.

        :param model: A model object which provides a `step` function to
            generate the system rollouts.
        :type model: models.base.BaseModel
        :param state: The initial state of the system.
        :type state: torch.Tensor
        :param ext_actions: A matrix of shape `n_actions` x `n_pol` x `hz_len` x
            `dim_a` action sequences.
        :type actions: torch.Tensor
        :returns: A tuple of (actions, states, eps) for `n_actions` * `tf.pts`
            rollouts.
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        if ext_actions is None:
            eps = self.a_dist.sample(
                sample_shape=[self.n_actions, self.n_pol, self.hz_len]
            )
            # apply eps to all policies
            actions = torch.add(eps, self.a_mat)
        else:
            actions = ext_actions
            # create eps from the difference between external actions and planned a_seq
            eps = torch.add(actions, -self.a_seq)

        # Create a matrix to store all sigma points for each trajectory and for
        # each step, last dim is the number of sigma points
        try:
            cov = params_dist.covariance_matrix
            mean = params_dist.mean
        except AttributeError:
            try:
                cov = params_dist.variance.diag()
                mean = params_dist.mean
            except AttributeError:  # for interfacing with BayesSim MoG
                idx = params_dist.a.argmax()
                cov = params_dist.xs[idx].S
                mean = params_dist.xs[idx].m
        params_sp = self._tf.compute_sigma_points(mean, cov)

        # prepare tensors for batch rollout
        # when using sigma points, the number of trajectory is effectively
        # `n_actions` * `tf.pts`
        # repeat `tf.pts` times each sequence of actions
        actions_sp = actions.repeat(1, 1, self._tf.pts, 1).reshape(
            -1, self.hz_len, self.dim_a
        )
        # repeat `n_actions` times the `params_sp` tensor, so every action is applied
        # to each sigma point
        params_dict = model.params_to_dict(
            params_sp.T.repeat(self.n_actions * self.n_pol, 1)
        )
        states_sp = state.expand(
            self.n_actions * self._tf.pts * self.n_pol, 1, self.dim_s
        ).clone()

        # generate rollouts
        for t in range(self.hz_len):
            states_sp = torch.cat(
                [
                    states_sp,
                    model.step(
                        states_sp[:, t], actions_sp[:, t], params_dict
                    ).unsqueeze(1),
                ],
                dim=1,
            )

        # restore dims of state_sp
        states_sp = states_sp.reshape(
            self.n_actions * self._tf.pts, self.n_pol, self.hz_len + 1, self.dim_s,
        )
        params = model.dict_to_params(params_dict).reshape(
            *actions.shape[:2], self._tf.pts, -1
        )
        params_log_p = params_dist.log_prob(params)
        # Sum the log probability of each parameter and average using tf
        # weights. Same as taking the probability of the mean for each param.
        params_log_p = params_log_p @ self._tf.loc_weights
        return states_sp, actions, eps, params_log_p

    def _compute_cost(self, states, actions, debug=False):
        """Estimate trajectories cost.

        :param states: A tensor with the states of each trajectory.
        :type states: torch.Tensor
        :param eps: A tensor with the difference of the current planned action
            sequence and the actions on each trajectory.
        :type eps: torch.Tensor
        :returns: A tensor with the costs for the given trajectories.
        :rtype: torch.Tensor
        """
        # Dims are n_params, n_actions, n_pol, hz_len, dim_s/dim_a
        x_vec = states[..., :-1, :].reshape(-1, self.dim_s)
        x_final = states[..., -1, :].reshape(-1, self.dim_s)
        a_vec = actions.reshape(-1, self.dim_a)
        inst_costs = self.inst_cost_fn(x_vec, a_vec, n_pol=self.n_pol, debug=debug)
        term_costs = self.term_cost_fn(x_final, n_pol=self.n_pol, debug=debug)

        if self._tf is not None:  # TODO: should we use log here?
            # Weight trajectories using UTF sigma weights
            inst_costs = torch.matmul(
                inst_costs.view(-1, self._tf.pts), self._tf.loc_weights
            )
            term_costs = torch.matmul(
                term_costs.view(-1, self._tf.pts), self._tf.loc_weights
            )
            inst_costs = inst_costs.view(self.n_actions, self.n_pol, self.hz_len)
            inst_costs = inst_costs.sum(dim=-1)  # sum over control horizon
            term_costs = term_costs.view(self.n_actions, self.n_pol)
            state_cost = inst_costs + term_costs
        else:
            inst_costs = inst_costs.view(
                self.n_params, self.n_actions, self.n_pol, self.hz_len
            )
            inst_costs = inst_costs.sum(dim=-1)  # sum over control horizon
            term_costs = term_costs.view(self.n_params, self.n_actions, self.n_pol)
            state_cost = (inst_costs + term_costs).mean(0)  # avg costs over params

        # To compute ctrl_costs in a single batch we first compute the term
        # a_mat * a_pre (n_pol x hz_len x dim_a), the result is then
        # contracted over hz_len and dim_a using tensordot. Finally, the
        # control cost cost is computed by taking the trace of the resulting
        # n_pol x n_actions x n_actions matrices over the last 2 dimensions.
        # The result is dim n_pol x n_actions.
        eps = torch.add(
            actions[0], -self.a_seq
        )  # actions are equal over each param batch
        ctrl_costs = self.a_reg * torch.tensordot(
            -eps, (self.a_mat @ self.a_pre), dims=([-2, -1], [-2, -1])
        )
        ctrl_costs = ctrl_costs.diagonal(dim1=-2, dim2=-1)

        return state_cost + ctrl_costs

    def forward(self, state, model, params_dist=None, ext_actions=None, debug=False):
        """Updates the control sequence and the predicted cost.

        ..note:
            This will *not* override the current planned actions, but update
            the plan of each individual controller and their weight (i.e.
            `a_mat` and `a_mix`).

        :param model: A model with a `step(states, actions, params)` function
            to compute the next state for a set of trajectories.
        :type model: models.base.BaseModel
        :param state: A with the system initial state.
        :type state: torch.Tensor
        :param actions: Actions sampled outside the controller scope. Must have
            shape `n_actions` x `n_pol` x `hz_len` x `dim_a`. If `None`, actions
            are sampled by the controller.
        :type actions: torch.Tensor
        :returns: A tuple of `(expected cost, states, actions, weights)` with
            the expected cost of the new actions and the sampled rollouts.
        :rtype: (float, torch.Tensor, torch.Tensor, torch.Tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float)
        if self._tf:
            states, actions, eps, params_dict = self._sigma_rollout(
                state, model, params_dist, ext_actions
            )
        else:
            states, actions, eps, params_dict = self._rollout(
                state, model, params_dist, ext_actions
            )
        # costs = self._compute_cost(states, eps)  # raw costs
        costs = self._compute_cost(states, actions, debug=debug)  # raw costs
        beta = costs.min()
        log_costs = -1 * (costs - beta) / self.temp
        # log normalising constant
        eta = log_costs.logsumexp(0)
        # log costs, tensor of size n_actions
        omega = log_costs - eta
        # Contract n_pol with computed weights
        delta = torch.tensordot(omega.exp().T, eps, dims=1)
        # Result is a n_pol x n_pol matrix of a_seq, we need only the main
        # diagonal (i.e. not the cross product between n_pol).
        # Use permute as torch appends the new dim with diagonal elements to the end
        # of the tensor.
        self.a_mat += delta.detach().diagonal().permute(2, 0, 1)
        self.a_mix = (eta - eta.logsumexp(0)).detach().exp()
        return costs, states, actions, omega.exp(), params_dict

    def step(self, strategy="argmax", steps=1, ext_actions=None):
        """Compute a new action sequence based on each individual controller,
        returns a given number of actions, and roll the sequence accordingly.
        """
        if strategy == "argmax":
            self.a_seq = self.a_mat[self.a_mix.argmax()]
        elif strategy == "average":
            self.a_seq = (self.a_mat.T @ self.a_mix).T
        elif strategy == "external" and ext_actions is not None:
            self.a_seq = ext_actions.clone()
        else:
            raise ValueError("Invalid value for strategy.")
        # Clip control actions before sending to actuators
        for idx, (min_a, max_a) in enumerate(zip(self.min_a, self.max_a)):
            self.a_seq[..., idx].clamp_(min_a, max_a)
        next_actions = self.a_seq[:steps]
        # Roll action sequence and matrix and initialize to neutral value
        self.a_seq = self.a_seq.roll(shifts=-steps, dims=0)
        self.a_seq[-steps:] = 0
        self.a_mat = self.a_mat.roll(shifts=-steps, dims=1)
        self.a_mat[:, -steps:] = 0
        return next_actions
