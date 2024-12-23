from abc import ABC, abstractmethod
import torch

class AbstractLearner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calc_loss(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    def train_step(self, batch) -> float:
        loss = self.calc_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer
