from abc import  ABC, abstractmethod
import torch

class AbstractLearner(ABC):

    def __init__(self):
        pass


    @abstractmethod
    def calc_loss(self, sample):
        pass
    


    def train_step(self):    # one step of the train --> no epoch idea
        sample = self.sample_data()
        loss = self.calc_loss(sample)
        self.update_model(loss)
        return loss.item()



    def update_model(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()