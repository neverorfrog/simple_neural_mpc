from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from simple_neural_mpc.robots.robot import Robot
from simple_neural_mpc.simulation.trajectory import Trajectory
from simple_neural_mpc.utils.fancy_vector import FancyVector


class Controller(ABC):
    """Controller Class"""

    def __init__(self):
        pass

    @abstractmethod
    def command(
        self, robot: Robot, reference: Trajectory, t: float
    ) -> Tuple[FancyVector, FancyVector, np.ndarray, np.ndarray]:
        """Compute the control actions
        Returns:
            (np.array): control actions
        """
        pass
