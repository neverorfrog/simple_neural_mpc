from abc import ABC, abstractmethod


class AbstractLearner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calc_loss(self, batch):
        pass

    def train_step(self, batch):
        loss = self.calc_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
