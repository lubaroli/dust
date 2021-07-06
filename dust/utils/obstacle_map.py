import os.path as osp
import random
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch

from .helper import from_np
from .obstacle import ObstacleRectangle


class ObstacleMap:
    """
    Generates an occupancy grid.
    """

    def __init__(self, map_dim, cell_size):

        assert map_dim[0] % 2 == 0
        assert map_dim[1] % 2 == 0

        cmap_dim = [0, 0]
        cmap_dim[0] = ceil(map_dim[0] / cell_size)
        cmap_dim[1] = ceil(map_dim[1] / cell_size)

        self.map = np.zeros(cmap_dim)
        self.cell_size = cell_size

        # Map center (in cells)
        self.origin_xi = int(cmap_dim[0] / 2)
        self.origin_yi = int(cmap_dim[1] / 2)

        self.x_dim, self.y_dim = self.map.shape
        x_range = self.cell_size * self.x_dim
        y_range = self.cell_size * self.y_dim
        self.xlim = [-x_range / 2, x_range / 2]
        self.ylim = [-y_range / 2, y_range / 2]

        self.c_offset = torch.Tensor([self.origin_xi, self.origin_yi])

    def convert_map(self):
        self.map_torch = from_np(self.map)
        return self.map_torch

    def plot(self, save_dir=None, filename="obst_map.png"):
        plt.figure()
        plt.imshow(self.map.T)
        plt.gca().invert_yaxis()
        plt.show()
        if save_dir is not None:
            plt.savefig(osp.join(save_dir, filename))

    def get_xy_grid(self):
        xv, yv = torch.meshgrid(
            [
                torch.linspace(self.xlim[0], self.xlim[1], self.x_dim),
                torch.linspace(self.ylim[0], self.ylim[1], self.y_dim),
            ]
        )
        xy_grid = torch.stack((xv, yv), dim=2)
        return xy_grid

    def get_collisions(self, X):
        """
        Checks for collision in a batch of trajectories using the generated
        occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weight: weight on obstacle cost, float tensor.
        :param X: Tensor of trajectories, of shape
         (batch_size, traj_length, position_dim)
        :return: collision cost on the trajectories
        """

        # Convert traj. positions to occupancy indices
        X_occ = X * (1 / self.cell_size) + self.c_offset
        X_occ = X_occ.floor()

        X_occ = X_occ.type(torch.LongTensor)

        # Project out-of-bounds locations to axis
        X_occ[..., 0] = X_occ[..., 0].clamp(0, self.map.shape[0] - 1)
        X_occ[..., 1] = X_occ[..., 1].clamp(0, self.map.shape[1] - 1)

        # Collisions
        try:
            collision_vals = self.map_torch[X_occ[..., 0], X_occ[..., 1]]
        except Exception as e:
            print(e)
            print(X_occ)
            print(X_occ.clamp(0, self.map.shape[0] - 1))
        return collision_vals


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return ceil(n * multiplier) / multiplier


def get_obst_preset(preset_name, obst_width=2):
    w = obst_width
    if preset_name == "staggered_3-2-3":
        obst_params = [
            [-4.0, 4.0, w, w],
            [0.0, 4.0, w, w],
            [4.0, 4.0, w, w],
            [-6, 0, w, w],
            [-2, 0, w, w],
            [2, 0, w, w],
            [6, 0, w, w],
            [-4.0, -4.0, w, w],
            [0.0, -4.0, w, w],
            [4.0, -4.0, w, w],
        ]

    elif preset_name == "staggered_4-3-4-3-4":
        obst_params = [
            [-6, 6, w, w],
            [-2.0, 6, w, w],
            [2.0, 6, w, w],
            [6, 6, w, w],
            [-4.0, 3, w, w],
            [0.0, 3, w, w],
            [4.0, 3, w, w],
            [-6, 0, w, w],
            [-2.0, 0, w, w],
            [2.0, 0, w, w],
            [6, 0, w, w],
            [-4, -3, w, w],
            [0.0, -3, w, w],
            [4, -3, w, w],
            [-6, -6, w, w],
            [-2, -6, w, w],
            [2, -6, w, w],
            [6, -6, w, w],
        ]

    elif preset_name == "grid_3x3":
        s = 5  # spacing
        obst_params = [
            [-s, s, w, w],
            [0.0, s, w, w],
            [s, s, w, w],
            [-s, 0, w, w],
            [0, 0, w, w],
            [s, 0, w, w],
            [-s, -s, w, w],
            [0.0, -s, w, w],
            [s, -s, w, w],
        ]
    elif preset_name == "grid_4x4":
        s = 4
        obst_params = [
            [-s * 3 / 2, s * 3 / 2, w, w],
            [-s * 1 / 2, s * 3 / 2, w, w],
            [s * 1 / 2, s * 3 / 2, w, w],
            [s * 3 / 2, s * 3 / 2, w, w],
            [-s * 3 / 2, s / 2, w, w],
            [-s * 1 / 2, s * 1 / 2, w, w],
            [s * 1 / 2, s * 1 / 2, w, w],
            [s * 3 / 2, s * 1 / 2, w, w],
            [-s * 3 / 2, -s * 1 / 2, w, w],
            [-s * 1 / 2, -s * 1 / 2, w, w],
            [s * 1 / 2, -s * 1 / 2, w, w],
            [s * 3 / 2, -s * 1 / 2, w, w],
            [-s * 3 / 2, -s * 3 / 2, w, w],
            [-s * 1 / 2, -s * 3 / 2, w, w],
            [s * 1 / 2, -s * 3 / 2, w, w],
            [s * 3 / 2, -s * 3 / 2, w, w],
        ]

    elif preset_name == "grid_6x6":
        w = obst_width
        s = 3
        obst_params = [
            [-s * 5 / 2, s * 5 / 2, w, w],
            [-s * 3 / 2, s * 5 / 2, w, w],
            [-s * 1 / 2, s * 5 / 2, w, w],
            [s * 1 / 2, s * 5 / 2, w, w],
            [s * 3 / 2, s * 5 / 2, w, w],
            [s * 5 / 2, s * 5 / 2, w, w],
            [-s * 5 / 2, s * 3 / 2, w, w],
            [-s * 3 / 2, s * 3 / 2, w, w],
            [-s * 1 / 2, s * 3 / 2, w, w],
            [s * 1 / 2, s * 3 / 2, w, w],
            [s * 3 / 2, s * 3 / 2, w, w],
            [s * 5 / 2, s * 3 / 2, w, w],
            [-s * 5 / 2, s / 2, w, w],
            [-s * 3 / 2, s / 2, w, w],
            [-s * 1 / 2, s * 1 / 2, w, w],
            [s * 1 / 2, s * 1 / 2, w, w],
            [s * 3 / 2, s * 1 / 2, w, w],
            [s * 5 / 2, s * 1 / 2, w, w],
            [-s * 5 / 2, -s * 1 / 2, w, w],
            [-s * 3 / 2, -s * 1 / 2, w, w],
            [-s * 1 / 2, -s * 1 / 2, w, w],
            [s * 1 / 2, -s * 1 / 2, w, w],
            [s * 3 / 2, -s * 1 / 2, w, w],
            [s * 5 / 2, -s * 1 / 2, w, w],
            [-s * 5 / 2, -s * 3 / 2, w, w],
            [-s * 3 / 2, -s * 3 / 2, w, w],
            [-s * 1 / 2, -s * 3 / 2, w, w],
            [s * 1 / 2, -s * 3 / 2, w, w],
            [s * 3 / 2, -s * 3 / 2, w, w],
            [s * 5 / 2, -s * 3 / 2, w, w],
            [-s * 5 / 2, -s * 5 / 2, w, w],
            [-s * 3 / 2, -s * 5 / 2, w, w],
            [-s * 1 / 2, -s * 5 / 2, w, w],
            [s * 1 / 2, -s * 5 / 2, w, w],
            [s * 3 / 2, -s * 5 / 2, w, w],
            [s * 5 / 2, -s * 5 / 2, w, w],
        ]

    elif preset_name == "single_centred":
        obst_params = [[0, 0, w, w]]

    else:
        raise IOError("Obstacle preset not supported: ", preset_name)
    return obst_params


def random_rect(xlim=(0, 0), ylim=(0, 0), width=2, height=2):
    """
    Generates a rectangular obstacle object, with random location and dimensions.
    """
    cx = random.uniform(xlim[0], xlim[1])
    cy = random.uniform(ylim[0], ylim[1])
    return ObstacleRectangle(cx, cy, width, height,)


def save_map_image(obst_map=None, start_pts=None, goal_pts=None, dir="."):
    try:
        plt.imshow(obst_map.T, cmap="gray")
        if start_pts is not None:
            for pt in start_pts:
                plt.plot(pt[0], pt[1], ".g")
        if goal_pts is not None:
            for pt in goal_pts:
                plt.plot(pt[0], pt[1], ".r")
        plt.gca().invert_yaxis()
        plt.savefig("{}/obst_map.png".format(dir))
    except Exception as err:
        print("Error: could not save map.")
        print(err)
    return


def generate_obstacle_map(
    map_dim=(10, 10),
    obst_list=[],
    cell_size=1.0,
    start_pts=None,
    goal_pts=None,
    random_gen=False,
    num_obst=0,
    rand_xy_limits=None,
    rand_shape=[2, 2],
    map_type=None,
    plot=False,
    delta=0.5,
    sigma=0.5,
):

    """
    Args
    ---
    map_dim : (int,int)
        2D tuple containing dimensions of obstacle/occupancy grid.
        Treat as [x,y] coordinates. Origin is in the center.
        ** Dimensions must be an even number. **
    cell_sz : float
        size of each square map cell
    obst_list : [(cx_i, cy_i, width, height)]
        List of obstacle param tuples
    start_pts : float
        Array of x-y points for start configuration.
        Dim: [Num. of points, 2]
    goal_pts : float
        Array of x-y points for target configuration.
        Dim: [Num. of points, 2]
    seed : int or None
    random_gen : bool
        Specify whether to generate random obstacles. Will first generate obstacles
        provided by obst_list, then add random obstacles until number specified by
        num_obst.
    num_obst : int
        Total number of obstacles
    rand_limit: [[float, float],[float, float]]
        List defining x-y sampling bounds [[x_min, x_max], [y_min, y_max]]
    rand_shape: [float, float]
        Shape [width, height] of randomly generated obstacles.
    """

    # Make occupancy grid
    obst_map = ObstacleMap(map_dim, cell_size)

    num_fixed = len(obst_list)
    for param in obst_list:
        cx, cy, width, height = param
        rect = ObstacleRectangle(cx, cy, width, height,)
        rect._add_to_map(obst_map)

    # Add obstacles to borders of the map
    for limit in obst_map.xlim:
        rect = ObstacleRectangle(
            limit, 0, 4 * obst_map.cell_size, obst_map.ylim[1] - obst_map.ylim[0],
        )
        rect._add_to_map(obst_map)
    for limit in obst_map.ylim:
        rect = ObstacleRectangle(
            0, limit, obst_map.xlim[1] - obst_map.xlim[0], 4 * obst_map.cell_size,
        )
        rect._add_to_map(obst_map)
    # Add random obstacles
    if random_gen:
        # random.seed(seed)
        assert num_fixed <= num_obst, (
            "Total number of obstacles must be greater than or equal to number"
            " specified in obst_list"
        )
        xlim = rand_xy_limits[0]
        ylim = rand_xy_limits[1]
        width = rand_shape[0]
        height = rand_shape[1]
        for _ in range(num_obst - num_fixed + 1):
            num_attempts = 0
            max_attempts = 25
            while num_attempts <= max_attempts:
                rect = random_rect(xlim, ylim, width, height)

                # Check validity of new obstacle
                valid = rect._obstacle_collision_check(obst_map)
                # rect._point_collision_check(obst_map,start_pts) & \
                # rect._point_collision_check(obst_map,goal_pts)

                if valid:
                    # Add to Map
                    rect._add_to_map(obst_map)
                    obst_list.append(
                        [rect.center_x, rect.center_y, rect.width, rect.height]
                    )
                    break

                if num_attempts == max_attempts:
                    print("Obstacle generation: Max. number of attempts reached. ")
                    print(
                        "Total num. obstacles: {}.  Num. random obstacles: {}.".format(
                            len(obst_list), len(obst_list) - num_fixed
                        )
                    )

                num_attempts += 1

    obst_map.convert_map()

    # Fit mapping model
    if map_type == "direct":
        return obst_map
    else:
        raise IOError('Map type "{}" not recognized'.format(map_type))


def generate_random_obstacle_map(
    map_dim=(256, 256), num_obst=0, start_pts=None, goal_pts=None,
):
    """
    Args
    ---
    map_dim : (int,int)7
        2D tuple containing dimensions of obstacle/occupancy grid.
        Treat as [x,y] coordinates, with origin at bottom-left corner.
    num_obst : int
        Number of obstacles
    start_pts : float
        Array of x-y points for start configuration.
        Dim: [Num. of points, 2]
    goal_pts : float
        Array of x-y points for target configuration.
        Dim: [Num. of points, 2]
    """

    w_min = 30
    w_max = 40
    h_min = 30
    h_max = 40

    obst_map = np.zeros(map_dim)

    while True:
        for i in range(num_obst):
            valid = False
            while not valid:
                rect = random_rect(map_dim, w_min, w_max, h_min, h_max,)
                valid = (
                    rect._obstacle_collision_check(obst_map)
                    & rect._point_collision_check(obst_map, start_pts)
                    & rect._point_collision_check(obst_map, goal_pts)
                )
            obst_map = rect._add_to_map(obst_map)
        if not np.any(obst_map > 1):
            break

    obst_map = 1 - obst_map  # Invert values
    save_map_image(obst_map, start_pts, goal_pts)
    return obst_map
