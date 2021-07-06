import torch


class BaseController:
    """Base class for module controllers."""

    def __init__(
        self,
        observation_space,
        action_space,
        hz_len,
        inst_cost_fn=None,
        term_cost_fn=None,
        init_actions=None,
    ):
        """Constructor for BaseController.

        :param observation_space: A Box object defining the action space.
        :type observation_space: dual_svmpc.utils.spaces.Space
        :param action_space: A Box object defining the action space.
        :type action_space: dual_svmpc.utils.spaces.Space
        :param hz_len: Number of time steps in the control horizon.
        :type hz_len: int
        :param init_actions:  A tensor of dimension `hz_len` by `dim_a`
            containing the initial set of control actions. If None, the sequence
            is initialized to zeros. Defaults to None.
        :type init_actions: torch.Tensor
        """
        self.hz_len = hz_len
        self.dim_s = observation_space.dim
        self.dim_a = action_space.dim
        self.min_a = action_space.low
        self.max_a = action_space.high
        if not init_actions:
            self.a_seq = torch.zeros((self.hz_len, self.dim_a))
        else:
            self.a_seq = init_actions

        # Need at least one cost function
        if inst_cost_fn is None and term_cost_fn is None:
            raise ValueError("Specify at least one cost function")

        def _null_cost_fn(state):
            """Pseudo-function to return a null cost.

            :param state: The current state of the system.
            :type state: torch.Tensor
            """
            return (state * 0).sum(-1)

        if inst_cost_fn is None:
            self.__inst_cost_fn = _null_cost_fn
        else:
            self.__inst_cost_fn = inst_cost_fn
        if term_cost_fn is None:
            self.__term_cost_fn = _null_cost_fn
        else:
            self.__term_cost_fn = term_cost_fn

    @property
    def inst_cost_fn(self):
        return self.__inst_cost_fn

    @property
    def term_cost_fn(self):
        return self.__term_cost_fn

    def roll(self, steps=1):
        """Shifts the control sequence forward in-place a given amount of
        `steps`.

        Shifts the current control sequence forward in-place by an amount of
        steps specified by the :code:`steps` parameter. The new rows are
        initialized to zeros.

        :param steps: The number of time steps to roll forward. Defaults to 1.

        """
        self.a_seq = torch.roll(self.a_seq, -steps, 0)
        self.a_seq[-steps:, :] = torch.zeros_like(self.a_seq[-steps:, :])

    @staticmethod
    def get_jacobian(inputs, outputs=None, func=None, create_graph=False):
        """Computes the Jacobian of outputs with respect to inputs.

        The Jacobian is computed either with respect to an output tensor or by
        computing it using the given mapping `func`. *Exactly* one of
        `outputs` or `func` arguments must be provided.

        :param inputs: A tensor for the input of param function, must have
        gradient information if transition function is not provided.
        :param outputs: A tensor with the output of `func`, must have gradient
                information. Defaults to None.
        :param func: A transition function to generate the output tensor.
            Defaults to None.
        :param create_graph: If True keeps the computational graph of the
            resulting Jacobian. Defaults to False.
        :returns: A tensor of shape (outputs.shape + inputs.shape) containing
            the Jacobian of outputs with respect to inputs.

        """
        if func is None and outputs is None:
            raise ValueError("Provide an output value or specify a function")
        if func is not None:
            inputs = inputs.squeeze().requires_grad_(True)
            outputs = func(inputs)
        jac = torch.zeros(outputs.view(-1).shape + inputs.view(-1).shape)
        for i, out in enumerate(outputs.view(-1)):
            row_i = torch.autograd.grad(
                out,
                inputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if row_i is None:
                # this element of output doesn't depend on the inputs, so leave
                # gradient 0
                continue
            else:
                jac[i] = row_i
        if create_graph:
            jac.requires_grad_()
        return jac

    @staticmethod
    def get_hessian(inputs, jacobian):
        """Computes the hessian of outputs with respect to inputs.

        :param inputs: The input tensor.
        :param jacobian: A jacobian matrix with a computational graph.
        :returns: A tensor of same shape as `jacobian` containing the Hessian of
                outputs with respect to inputs.

        """
        assert jacobian.requires_grad and inputs.requires_grad
        hes = torch.zeros_like(jacobian)
        for i, out in enumerate(jacobian):
            for j in range(inputs.shape[0]):
                e = torch.autograd.grad(
                    out[j], inputs, retain_graph=True, allow_unused=True
                )[0]
                if e is None:
                    # this element of output doesn't depend on the inputs, so
                    # leave gradient 0
                    continue
                else:
                    hes[i, j] = e[j]
        return hes

    def forward(self, model, state):
        raise NotImplementedError("Should be implemented by the subclass")
