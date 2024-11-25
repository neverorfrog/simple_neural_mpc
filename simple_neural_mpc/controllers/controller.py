from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from simple_neural_mpc.utils.fancy_vector import FancyVector


class Controller(ABC):
    """Controller Class"""

    def __init__(self):
        pass

    @abstractmethod
    def command(self, *args, **kwargs) -> Tuple[FancyVector, FancyVector, np.ndarray, np.ndarray]:
        """Compute the control actions
        Returns:
            (np.array): control actions
        """
        pass
