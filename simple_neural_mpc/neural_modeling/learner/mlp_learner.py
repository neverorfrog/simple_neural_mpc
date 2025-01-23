from enum import Enum

import lightning as L
import torch

from simple_neural_mpc.config.neural_config import (
    TrainerConfig as trainer_config,
)
from simple_neural_mpc.neural_modeling.dataset.tensor_dataset import TensorDataset
from simple_neural_mpc.neural_modeling.learner.mlp import MLP
from simple_neural_mpc.robots.robot import Robot


class Phase(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class MlpLearner(L.LightningModule):

    def __init__(self, robot: Robot):
        super(MlpLearner, self).__init__()
        self.robot = robot
        self.state_dim = len(robot.state)
        self.input_dim = len(robot.input)
        self.mlp = MLP(self.state_dim, self.input_dim, in_mpc=False, is_pinn = False)
        self.mse = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return self.mlp(x)

    def training_step(self, batch: TensorDataset, batch_idx: int) -> torch.Tensor:
        predictions = self.forward(batch.data)
        loss = self.mse(predictions, batch.labels)
        self.log(f"{Phase.TRAIN.value}/loss", loss, batch_size=batch.data.shape[0])
        return loss

    def validation_step(self, batch: TensorDataset, batch_idx: int) -> torch.Tensor:
        predictions = self.forward(batch.data)
        loss = self.mse(predictions, batch.labels)
        self.log(f"{Phase.VAL.value}/loss", loss, batch_size=batch.data.shape[0])
        return loss

    def test_step(self, batch: TensorDataset, batch_idx: int) -> torch.Tensor:
        predictions = self.forward(batch.data)
        loss = self.mse(predictions, batch.labels)
        self.log(f"{Phase.TEST.value}/loss", loss, batch_size=batch.data.shape[0])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=trainer_config.lr,
            weight_decay=trainer_config.weight_decay,
        )
