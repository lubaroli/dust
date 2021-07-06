import math

import torch

from ..utils.spaces import Box
from .base import BaseModel


class SkidSteerRobot(BaseModel):
    """
    This class implements a simplified kinematic model for a skid-steer robot, based on:

    Kozlowski, K., Pazderski, D. (2004). Modeling and control of a 4-wheel
    skid-steering mobile robot. International Journal of Applied Mathematics and
    Computer Science, 14(4), 477--496.

    """

    def __init__(
        self,
        delta_t,
        x_icr=0.2,
        wheel_radius=0.0625,
        axial_distance=0.475,
        min_wheel_speed=-0.5,
        max_wheel_speed=0.5,
        **kwargs
    ):
        """
        Constructor.

        :param delta_t: time step for state transitions
        :param x_icr: x coordinate of the instant centre of rotation in the robot's
        local frame
        :param wheel_radius: robot's wheel radius
        :param axial_distance: distance between the wheels on opposite sides of the
        robot
        """
        params_dict = {
            "x_icr": x_icr,
            "wheel_radius": wheel_radius,
            "axial_distance": axial_distance,
        }
        super(SkidSteerRobot, self).__init__(
            dt=delta_t, params_dict=params_dict, **kwargs
        )

        self.__observation_space = Box(
            dim=5, low=-float("inf"), high=float("inf"), dtype=torch.float,
        )
        self.__action_space = Box(
            dim=2, low=min_wheel_speed, high=max_wheel_speed, dtype=torch.float,
        )

    @property
    def observation_space(self):
        """The Skid Steer Robot observation space.

        :return: A space with the Skid Steer Robot observation space.
        :rtype: Box
        """
        return self.__observation_space

    @property
    def action_space(self):
        """The Skid Steer Robot action space.

        :return: A space with the Skid Steer Robot action space.
        :rtype: Box
        """
        return self.__action_space

    def step(self, states, actions, params_dict):
        """
        Returns the next state for a given action and parameter setting.

        :param states: a N-by-3 tensor of the initial state (x,y,theta)
        :param actions: a N-by-2 tensor of control actions
        (right_wheel_speed,left_wheel_speed) [rot/s]
        :param params: kinematic model parameters [x_icr,wheel_radius,axial_distance],
        an N-by-3 tensor
        """

        x, y, theta, v, omega = states.chunk(5, dim=1)
        if params_dict is not None:
            batch_params = self.params_dict.copy()
            for key in params_dict.keys():
                batch_params[key] = params_dict[key]
            x_icr, wheel_radius, axial_distance = batch_params.values()
        else:
            x_icr, wheel_radius, axial_distance = self.params_dict.values()

        right_speed, left_speed = actions.clone().chunk(2, dim=1)
        right_speed.clamp_(self.action_space.low[0], self.action_space.high[0])
        left_speed.clamp_(self.action_space.low[1], self.action_space.high[1])

        linear_speed = (right_speed + left_speed) * math.pi * wheel_radius
        angular_speed = (
            (right_speed - left_speed) * 2 * math.pi * wheel_radius / axial_distance
        )

        forward_shift = linear_speed * self.dt
        lateral_shift = -angular_speed * x_icr * self.dt

        new_x = x + forward_shift * torch.cos(theta) - lateral_shift * torch.sin(theta)
        new_y = y + forward_shift * torch.sin(theta) + lateral_shift * torch.cos(theta)

        new_theta = theta + angular_speed * self.dt

        # reshape to make sure speeds are the same length even without sampling params
        new_state = torch.cat(
            [
                new_x,
                new_y,
                new_theta,
                linear_speed.expand_as(x),
                angular_speed.expand_as(x),
            ],
            dim=1,
        )

        return new_state
