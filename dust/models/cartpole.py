import math

import torch
from .base import BaseModel
from ..utils.spaces import Box


class CartPoleModel(BaseModel):
    """Model for the classic control problem of balancing a pole attached to
    a cart with an un-actuated joint using binary actions.

    Refer to A. G. Barto, R. S. Sutton, and C. W. Anderson, “Neuron-like
    adaptive elements that can solve difficult learning control problems,”
    IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-13,
    pp. 834–846, Sept./Oct. 1983.

    Observation space:
        Type: Box(4)

        +-----+----------------------+---------+--------+
        | Num | Observation          | Min     | Max    |
        +=====+======================+=========+========+
        | 0   | Cart Position        | -4.8    | 4.8    |
        +-----+----------------------+---------+--------+
        | 1   | Cart Velocity        | -Inf    | Inf    |
        +-----+----------------------+---------+--------+
        | 2   | Pole Angle           | -24 deg | 24 deg |
        +-----+----------------------+---------+--------+
        | 3   | Pole Velocity At Tip | -Inf    | Inf    |
        +-----+----------------------+---------+--------+

    Actions space:
        Type: Box(1)

        +-----+-----------+-----+-----+
        | Num | Action    | Min | Max |
        +=====+===========+=====+=====+
        | 0   | Push cart | -1  | +1  |
        +-----+-----------+-----+-----+
    """

    def __init__(
        self,
        g=9.8,
        f_mag=10.0,
        mass_cart=1.0,
        mass_pole=0.1,
        length=1.0,
        mu_c=0.5e-3,
        mu_p=2e-6,
        **kwargs
    ):
        """Constructor for CartPoleModel.

        :param g: Gravity in `m/s^2`. Defaults to `9.8`.
        :type g: float
        :param f_mag: Magnitude of the force applied to the cart in `N`.
            Defaults to `10.0`.
        :type f_mag: float
        :param mass_cart: Mass of the cart in `kg`. Defaults to `1.0`.
        :type mass_cart: float
        :param mass_pole: Mass of the pole in `kg`. Defaults to `0.1`.
        :type mass_pole: float
        :param length: Pole length in `m`. Defaults to `1.0`.
        :type length: float
        :param mu_c: Cart friction constant. Defaults to `0.5e(-3)`.
        :type mu_c: float
        :param mu_p: Pole friction constant. Defaults to `2e(-6)`.
        :type mu_p: float
        :key dt: Duration of each discrete update in s. (default: 0.05)
        :type dt: float
        :key uncertain_params: A tuple containing the uncertain parameters of
            the forward model. Is used as keys for assigning sampled parameters
            from the `params_dist` function. Defaults to None.
        :key params_dist: A distribution to sample parameters for the forward
            model.
        :type kwargs: tuple or torch.distributions.distribution.Distribution
        """
        params_dict = {
            "g": g,
            "mass_cart": mass_cart,
            "mass_pole": mass_pole,
            "length": length,
            "mu_c": mu_c,
            "mu_p": mu_p,
            "f_mag": f_mag,
        }
        super().__init__(params_dict=params_dict, **kwargs)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = torch.tensor(
            [
                self.x_threshold * 2,
                float("Inf"),
                self.theta_threshold_radians * 2,
                float("Inf"),
            ]
        )

        self.__action_space = Box(dim=1, low=-1, high=1, dtype=torch.float32)
        self.__observation_space = Box(4, -high, high, dtype=torch.float32)

    @property
    def observation_space(self):
        """The Pendulum observation space.

        :return: A space with the Pendulum observation space.
        :rtype: Box
        """
        return self.__observation_space

    @property
    def action_space(self):
        """The Pendulum action space.

        :return: A space with the Pendulum action space.
        :rtype: Discrete
        """
        return self.__action_space

    def step(self, states, actions, params_dict=None):
        """Receives tensors of current states and actions and computes the
        states for the subsequent timestep. If sampled parameters are provided,
        these must be used, otherwise revert to mode of distribution over the
        uncertain parameters.

        Must be bounded by observation and action spaces.

        :param states: A tensor containing the current states of one or multiple
            trajectories.
        :type states: torch.Tensor
        :param actions: A tensor containing the next planned actions of one or
            multiple trajectories.
        :type actions: torch.Tensor
        :param sampled_params: A tensor containing samples for the uncertain
            system parameters. Note that the number of samples must be either 1
            or the number of trajectories. If 1, a single sample is used for all
            trajectories, otherwise use one sample per trajectory.
        :type sampled_params: dict
        :returns: A tensor with the next states of one or multiple trajectories.
        :rtype: torch.Tensor
        """
        dt = self.dt
        x, x_d, theta, theta_d = states.chunk(4, dim=1)
        if params_dict is not None:
            params = self.__params_dict.copy()
            for key in params_dict.keys():
                params[key] = params_dict[key]
            g, m_c, m_p, length, mu_c, mu_p, f_mag = params.values()
        else:
            g, m_c, m_p, length, mu_c, mu_p, f_mag = self.__params_dict.values()
        # even though gym CartPole is binary, since our forward model is
        # continuous we have to center actions at 0 and clip on [-1, +1]
        acts = torch.clamp(actions, min=-1, max=1) * f_mag

        mass = m_c + m_c  # total mass
        pm = m_p * length  # pole-mass
        cart_friction = mu_c * x_d.sign()
        pole_friction = (mu_p * theta_d) / pm
        factor = (acts + pm * theta.sin() * theta_d ** 2 - cart_friction) / mass
        tdd_num = g * theta.sin() - theta.cos() * factor - pole_friction
        tdd_den = length * (4.0 / 3 - (m_p * theta.cos() ** 2) / mass)
        theta_dd = tdd_num / tdd_den

        x_dd = factor - pm * theta_dd * torch.cos(theta) / mass
        delta = torch.cat([x_d, x_dd, theta_d, theta_dd], dim=1) * dt
        return states + delta
