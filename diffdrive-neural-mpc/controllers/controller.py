from abc import ABC, abstractmethod


class Controller(ABC):
    """Controller Class"""

    def __init__(self):
        pass

    @abstractmethod
    def command(self, *args, **kwargs):
        """Compute the control actions
        Returns:
            (np.array): control actions
        """
        pass
