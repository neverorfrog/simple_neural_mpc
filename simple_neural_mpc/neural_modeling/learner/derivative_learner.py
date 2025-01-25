from enum import Enum

import lightning as L
from matplotlib import pyplot as plt
import numpy as np
import torch

from simple_neural_mpc.config.neural_config import (
    TrainerConfig as trainer_config,
)
from simple_neural_mpc.neural_modeling.dataset.tensor_dataset import TensorDataset
from simple_neural_mpc.neural_modeling.learner.mlp import MLP, Sine
from simple_neural_mpc.config.mpc_config import MPCConfig


class Phase(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DerivativeLearner(L.LightningModule):

    def __init__(self, state_dim: int, input_dim: int, in_mpc: bool = True):
        super(DerivativeLearner, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.mlp = MLP(
            self.state_dim,
            self.input_dim,
            activation = torch.nn.Tanh(),
            predicts_state=False,
            in_mpc=in_mpc,
            is_highway=False,
        )
        self.mse = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return self.mlp(x)

    def training_step(self, batch: TensorDataset, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch, Phase.TRAIN)
        return loss

    def validation_step(self, batch: TensorDataset, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch, Phase.VAL)
        return loss

    def test_step(self, batch: TensorDataset, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch, Phase.TEST)
        return loss
    

    def compute_loss(self, batch: TensorDataset, phase: Phase) -> torch.Tensor:
        predictions = self.forward(batch.data)
        loss = self.mse(predictions, batch.labels)
        self.log(
            f"{phase.value}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.data.shape[0],
        )
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=trainer_config.lr,
            weight_decay=trainer_config.weight_decay,
        )
        
    def test_traj(self, test_data: TensorDataset) -> torch.Tensor:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        test_data.data = test_data.data[:, :, :-1]
        predictions = self.forward(test_data.data).detach().numpy()
        
        print(predictions.shape)

        random_idx = np.random.randint(0, test_data.data.shape[0])
        ground_truth = test_data.data[random_idx][:, :3]

        random_prediction = predictions[random_idx]
        predicted_traj = []
        for i in range(random_prediction.shape[0]):
            predicted_state = ground_truth[i] + random_prediction[i] * MPCConfig.dt
            predicted_traj.append(predicted_state)

        predicted_traj = np.array(predicted_traj)

        ax.plot(ground_truth[1:, 0], ground_truth[1:, 1], 'r')
        ax.plot(predicted_traj[:-1, 0], predicted_traj[:-1, 1], 'b')
