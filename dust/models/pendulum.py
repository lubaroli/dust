import math

import torch

from ..utils.spaces import Box
from .base import BaseModel


class PendulumModel(BaseModel):
    """Model for a fixed joint 1-DOF inverted pendulum.

    For more information refer to OpenAI Gym environment at:
    https://gym.openai.com/envs/Pendulum-v0/
    """

    def __init__(self, g=9.8, mass=1.0, length=1.0, **kwargs):
        """Constructor for PendulumModel.

        :param g: Gravity in m/s^2. Defaults to 9.8.
        :param mass: Pendulum mass in kg. Defaults to 1.0.
        :param length: Pendulum length in metres. Defaults to 1.0.
        :key dt: Duration of each discrete update in s. Defaults to 0.05
        :type dt: float
        :key uncertain_params: A tuple containing the uncertain parameters of
            the forward model. Is used as keys for assigning sampled parameters
            from the `params_dist` function. Defaults to None.
        :key params_dist: A distribution to sample parameters for the forward
            model.
        :type kwargs: tuple or torch.distributions.distribution.Distribution
        """
        params_dict = {"g": g, "mass": mass, "length": length}
        super().__init__(params_dict=params_dict, **kwargs)
        self.__max_speed = 8.0
        self.__max_torque = 2.0
        bounds = torch.tensor([float("inf"), self.__max_speed])
        self.__observation_space = Box(
            dim=2, low=-bounds, high=bounds, dtype=torch.float
        )
        self.__action_space = Box(
            dim=1, low=-self.__max_torque, high=self.__max_torque, dtype=torch.float,
        )

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
        :rtype: Box
        """
        return self.__action_space

    def step(self, states, actions, params_dict=None):
        """Receives tensors of current states and actions and computes the
        states for the subsequent timestep. If sampled parameters are provided,
        these must be used, otherwise default model parameters are used.

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
        # Assigning states and params, keeping their dims
        theta, theta_d = states.clone().chunk(2, dim=-1)
        if params_dict is not None:
            batch_params = self.params_dict.copy()
            for key in params_dict.keys():
                batch_params[key] = params_dict[key]
            g, m, length = batch_params.values()
        else:
            g, m, length = self.params_dict.values()

        acts = actions.clamp(min=-self.__max_torque, max=self.__max_torque)
        theta_d = theta_d + dt * (
            -3 * g / (2 * length) * (theta + math.pi).sin()
            + 3.0 / (m * length ** 2) * acts
        )
        theta_d = theta_d.clamp(-self.__max_speed, self.__max_speed)
        theta = theta + theta_d * dt  # Use new theta_d
        return torch.cat((theta, theta_d), dim=-1)

    @staticmethod
    def get_obs(state):
        try:
            theta, theta_d = state.chunk(2, dim=1)
        except ValueError:
            raise ValueError("Dimension 1 of state tensor must be exactly 2.")
        return torch.cat([torch.cos(theta), torch.sin(theta), theta_d], dim=1)
