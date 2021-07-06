from abc import ABC, abstractmethod
from copy import deepcopy
from math import ceil

import numpy as np


class Obstacle(ABC):
    """
    Base 2D Obstacle class
    """

    def __init__(self, center_x, center_y):
        self.center_x = int(center_x)
        self.center_y = int(center_y)

    @abstractmethod
    def _obstacle_collision_check(self, obst_map):
        pass

    @abstractmethod
    def _point_collision_check(self, obst_map, pts):
        pass

    @abstractmethod
    def _add_to_map(self, obst_map):
        pass


class ObstacleRectangle(Obstacle):
    """
    Derived 2D rectangular Obstacle class
    """

    def __init__(self, center_x=0, center_y=0, width=None, height=None):
        super().__init__(center_x, center_y)
        self.width = width
        self.height = height

    def _obstacle_collision_check(self, obst_map):
        valid = True
        obst_map_test = self._add_to_map(deepcopy(obst_map))
        if np.any(obst_map_test.map > 1):
            valid = False
        return valid

    def _point_collision_check(self, obst_map, pts):
        valid = True
        if pts is not None:
            obst_map_test = self._add_to_map(np.copy(obst_map))
            for pt in pts:
                if obst_map_test[ceil(pt[0]), ceil(pt[1])] == 1:
                    valid = False
                    break
        return valid

    def _add_to_map(self, obst_map):
        # Convert dims to cell indices
        w = ceil(self.width / obst_map.cell_size)
        h = ceil(self.height / obst_map.cell_size)
        c_x = ceil(self.center_x / obst_map.cell_size)
        c_y = ceil(self.center_y / obst_map.cell_size)

        x_start = c_x - ceil(w / 2.0) + obst_map.origin_xi
        x_end = c_x + ceil(w / 2.0) + obst_map.origin_xi
        y_start = c_y - ceil(h / 2.0) + obst_map.origin_yi
        y_end = c_y + ceil(h / 2.0) + obst_map.origin_yi
        obst_map.map[x_start:x_end, y_start:y_end] = 1
        return obst_map
