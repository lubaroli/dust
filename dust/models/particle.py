import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from ..utils.obstacle_map import generate_obstacle_map, get_obst_preset
from ..utils.spaces import Box
from .base import BaseModel


class Particle(BaseModel):
    def __init__(
        self,
        mass=1.0,
        noise_std=torch.zeros(2),
        control_type="acceleration",
        cost_params=None,
        with_obstacle=False,
        obst_preset=None,
        obst_width=None,
        obst_params=None,
        map_size=None,
        map_type=None,
        map_cell_size=None,
        init_state=None,
        target_state=None,
        can_crash=False,
        max_speed=None,
        max_accel=None,
        verbose=False,
        deterministic=False,
        euler_steps=1,
        **kwargs,
    ):
        params_dict = {"mass": mass}
        super().__init__(params_dict=params_dict, **kwargs)
        self.__max_speed = (
            torch.tensor(float("inf")) if max_speed is None else max_speed
        )
        self.__max_acc = torch.tensor(float("inf")) if max_accel is None else max_accel
        if control_type == "velocity":
            bounds = torch.tensor([float("inf"), float("inf")])
            self.__observation_space = Box(
                dim=2, low=-bounds, high=bounds, dtype=torch.float
            )
            self.__action_space = Box(
                dim=2, low=-self.__max_speed, high=self.__max_speed, dtype=torch.float,
            )
        elif control_type == "acceleration":
            bounds = torch.tensor(
                [float("inf"), float("inf"), self.__max_speed, self.__max_speed,]
            )
            self.__observation_space = Box(
                dim=4, low=-bounds, high=bounds, dtype=torch.float
            )
            self.__action_space = Box(
                dim=2, low=-self.__max_acc, high=self.__max_acc, dtype=torch.float,
            )
        else:
            raise IOError('control_type "{}" not recognized'.format(control_type))
        if target_state is None:
            self.target = torch.zeros(self.observation_space.dim)
        else:
            self.target = torch.as_tensor(target_state)

        self.dyn_std = noise_std
        self.init_state = torch.as_tensor(init_state)
        self.euler_steps = euler_steps
        self._small_dt = self.dt / euler_steps
        self.control_type = control_type

        self.with_obstacle = with_obstacle
        self.can_crash = can_crash
        self.obst_params = obst_params

        self.map_cell_size = map_cell_size
        assert map_size[0] % 2 == 0
        assert map_size[1] % 2 == 0
        self.map_size = map_size
        self.cmap_size = [0, 0]
        self.cmap_size[0] = torch.as_tensor(map_size[0] / self.map_cell_size).ceil()
        self.cmap_size[1] = torch.as_tensor(map_size[1] / self.map_cell_size).ceil()
        # Map center (in cells)
        origin_xi = int(self.cmap_size[0] / 2)
        origin_yi = int(self.cmap_size[1] / 2)
        self.c_offset = torch.Tensor([origin_xi, origin_yi])

        self.verbose = verbose
        self.deterministic = deterministic

        self.init_cost_weights(cost_params)

        if self.with_obstacle:
            self.obst_params = get_obst_preset(obst_preset, obst_width,)
            self.obst_map = generate_obstacle_map(
                map_size, self.obst_params, map_cell_size, map_type=map_type,
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
        """
        Rolls out the dynamics and then evaluates the resultant trajectories.

        Parameters
        ----
        x0 : Tensor
            Start states, of shape [batch, state_samples, state_dim]
        Us : Tensor
            Control sequences, of shape [steps, batch, state_samples, control_dim]

        Returns
        -------
        costs : Tensor
            Trajectory costs, of shape [batch] or [batch, particles]
        X, U : Tensor
            Trajectory and control tensors
        """
        # Fetch environment params
        if params_dict is not None:
            batch_params = self.params_dict.copy()
            for key in params_dict.keys():
                batch_params[key] = params_dict[key]
            (m,) = batch_params.values()
        else:
            (m,) = self.params_dict.values()
        # Prepare action vector
        acts = actions.clone()
        if not self.deterministic:
            # Noise in control channel
            noise = self.dyn_std * torch.randn_like(acts)
            acts += noise
        if self.control_type == "acceleration":
            acts = torch.clamp(acts / m, min=-self.__max_acc, max=self.__max_acc)
        elif self.control_type == "velocity":
            acts.clamp_(min=-self.__max_speed, max=self.__max_speed)
        # Step and check for collisions
        x_dot = torch.cat((states[..., 2:], acts), dim=-1)
        if self.can_crash and self.with_obstacle:
            # Check if current state in collision. If so, particle has
            # "crashed" and remains in that position.
            collision_mask = self.obst_map.get_collisions(states[..., 0:2],).unsqueeze(
                -1
            )
            next_states = states + x_dot * self.dt * (1 - collision_mask)
        else:
            next_states = states + x_dot * self.dt
        # Enforce velocity bounds
        next_states[..., -2:].clamp_(min=-self.__max_speed, max=self.__max_speed)
        return next_states

    # def default_inst_cost(self, states, actions=0):
    # def default_inst_cost(self, states, actions=0, n_pol=0, hz_len=0, s_dim=0, a_dim=0):
    def default_inst_cost(self, states, actions=0, n_pol=0, debug=False):
        # Collision costs
        if self.with_obstacle:
            # Detect collisions using x and y position
            collision_vals = self.obst_map.get_collisions(states[..., 0:2])
            obst_cost = self.w_obs * collision_vals
        else:
            obst_cost = 0.0
        # Distance to goal
        delta_pos = states - self.target
        state_cost = torch.mul(delta_pos, delta_pos) * self.w_state
        control_cost = torch.mul(actions, actions) * self.w_ctrl

        if debug:
            print("\n Avg. instant. costs:")
            # print("control: ", control_cost.view(-1, hz_len, a_dim))
            # print("state: ", state_cost.view(-1, hz_len, s_dim))
            # print("obst: ", obst_cost.view(-1, hz_len, s_dim))
            print(
                "control: ", self.to_numpy(control_cost.view(n_pol, -1).sum(1).mean(0)),
            )
            print(
                "state: ", self.to_numpy(state_cost.view(n_pol, -1).sum(1).mean(0)),
            )
            print(
                "obst: ", self.to_numpy(obst_cost.view(n_pol, -1).sum(1).mean(0)),
            )

        return state_cost.sum(-1) + control_cost.sum(-1) + obst_cost

    # def default_term_cost(self, states):
    # def default_term_cost(self, states, n_pol=0, hz_len=0, s_dim=0, a_dim=0):
    def default_term_cost(self, states, n_pol=0, debug=False):

        # Collision costs
        if self.with_obstacle:
            # Detect collisions using x and y position
            collision_vals = self.obst_map.get_collisions(states[..., 0:2])
            obst_cost = self.w_obs * collision_vals
        else:
            obst_cost = 0.0

        # Distance to goal
        delta_pos = states - self.target
        state_cost = torch.mul(delta_pos, delta_pos) * self.w_term

        if debug:
            print("\n Avg. terminal costs:")
            print(
                "state: ", self.to_numpy(state_cost.view(n_pol, -1).sum(1).mean(0)),
            )
            print(
                "obst: ", self.to_numpy(obst_cost.view(n_pol, -1).sum(1).mean(0)),
            )
            print("\n")
        return state_cost.sum(-1) + obst_cost

    def render(
        self,
        states=None,
        rollouts=None,
        path=None,
        ax=None,
        cmap="Oranges",
        plot_goal=True,
        **plot_kwargs,
    ):
        ax = plt.gca() if ax is None else ax
        ax.set_xlim(0, self.cmap_size[0])
        ax.set_ylim(0, self.cmap_size[1])
        if self.with_obstacle:
            ax.imshow(self.obst_map.map.T, cmap=cmap)
        if plot_goal:
            # plot initial state
            ax.scatter(
                self.to_map_coord(self.init_state[:2])[0],
                self.to_map_coord(self.init_state[:2])[1],
                marker="o",
                color="r",
                s=200,
            )
            # plot target state
            ax.scatter(
                self.to_map_coord(self.target[:2])[0],
                self.to_map_coord(self.target[:2])[1],
                marker="*",
                color="r",
                s=200,
            )
        # plot state
        if states is not None:
            ax.plot(
                self.to_map_coord(states[..., :2])[..., 0],
                self.to_map_coord(states[..., :2])[..., 1],
                **plot_kwargs,
            )
        # plot rollouts
        if rollouts is not None:
            # rollouts dimensions are batch x n_policies x timestep x state
            map_rollouts = self.to_map_coord(rollouts[..., :2])
            n_pol = map_rollouts.shape[1]
            colour = iter(cm.rainbow(torch.linspace(0, 1, n_pol).numpy()))
            for policy in range(n_pol):
                ax.plot(
                    map_rollouts[:, policy, :, 0].T,
                    map_rollouts[:, policy, :, 1].T,
                    alpha=0.3,
                    color=next(colour),
                    linewidth=1,
                )

        if path is not None:
            if not path.parent.exists():
                path.parent.mkdir()
            plt.savefig(path)

        return ax

    def to_map_coord(self, coord_vec):
        assert coord_vec.shape[-1] == 2, "Coordinates must be 2-D."
        return self.c_offset + coord_vec / self.map_cell_size

    def init_cost_weights(self, params):
        if params is None:
            keys = [
                "w_qpos",
                "w_qvel",
                "w_qpos_T",
                "w_qvel_T",
                "w_ctrl",
                "w_obs",
            ]
            params = dict.fromkeys(keys, 1.0)

        ### State cost weights
        w_qpos = [params["w_qpos"]] * 2  # Cartesian dist to target
        w_qvel = [params["w_qvel"]] * 2

        if self.control_type == "velocity":
            self.w_state = torch.as_tensor(w_qpos)
        elif self.control_type == "acceleration":
            self.w_state = torch.as_tensor(w_qpos + w_qvel)

        ### Control cost weights
        self.w_ctrl = torch.as_tensor([params["w_ctrl"]] * self.action_space.dim)

        ### Terminal cost weights
        w_qpos_T = [params["w_qpos_T"]] * 2  # Cartesian dist to target
        w_qvel_T = [params["w_qvel_T"]] * 2

        if self.control_type == "velocity":
            self.w_term = torch.as_tensor(w_qpos_T)
        elif self.control_type == "acceleration":
            self.w_term = torch.as_tensor(w_qpos_T + w_qvel_T)

        ### Obstacle cost weights
        self.w_obs = torch.as_tensor([params["w_obs"]])

    def to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().clone().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            raise IOError("Invalid type for numpy conversion.")
