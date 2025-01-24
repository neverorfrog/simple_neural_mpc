from enum import Enum

import lightning as L
from matplotlib import pyplot as plt
import torch

from simple_neural_mpc.config.neural_config import (
    TrainerConfig as trainer_config,
)
from simple_neural_mpc.neural_modeling.dataset.tensor_dataset import TensorDataset
from simple_neural_mpc.neural_modeling.learner.mlp import MLP
from simple_neural_mpc.robots.unicycle import Unicycle
from simple_neural_mpc.config.neural_config import DatasetConfig
import numpy as np
from simple_neural_mpc.neural_modeling.learner.mlp import Sine
from torch.func import jacrev, vmap


class Phase(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class NextStateLearner(L.LightningModule):

    def __init__(self, state_dim, input_dim, is_pinn: bool = True):
        super(NextStateLearner, self).__init__()
        self.is_pinn = is_pinn
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.mlp = MLP(
            self.state_dim,
            self.input_dim,
            activation=torch.nn.Tanh(),
            in_mpc=False,
            is_pinn=True,
            is_highway=False,
        )
        self.mse = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return self.mlp(x)

    def training_step(self, batch: TensorDataset, batch_idx: int) -> torch.Tensor:
        return self.compute_loss(batch, Phase.TRAIN)

    def validation_step(self, batch: TensorDataset, batch_idx: int) -> torch.Tensor:
        return self.compute_loss(batch, Phase.VAL)

    def test_step(self, batch: TensorDataset, batch_idx: int) -> torch.Tensor:
        return self.compute_loss(batch, Phase.TEST)

    def compute_loss(self, batch: TensorDataset, phase: Phase) -> torch.Tensor:
        loss = 0
        loss += self.compute_imit_loss(batch, phase)
        if self.is_pinn:
            loss += self.compute_physics_loss(batch, phase)

        self.log(
            f"{phase.value}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.data.shape[0],
        )
        return loss

    def compute_imit_loss(self, batch: TensorDataset, phase: Phase) -> torch.Tensor:
        """
        Computes the imitation loss
        """
        predicted_states = self.forward(batch.data)  # [batch_size, traj_len, state_dim]
        imit_loss = self.mse(predicted_states, batch.labels)
        self.log(
            f"{phase.value}/imit_loss",
            imit_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.data.shape[0],
        )
        return imit_loss

    def compute_physics_loss(self, batch: TensorDataset, phase: Phase) -> torch.Tensor:
        """
        Computes the physics loss
        """

        sampled_input = torch.rand(1024, self.state_dim + self.input_dim + 1).to(
            self.device
        )
        sampled_input = (sampled_input * 10 + 5).float()

        # jacobian --> [batch, state_dim, (state_dim + input_dim + 1(time))]
        jac = vmap(jacrev(self.mlp))(sampled_input).squeeze()
        state_derivative_numeric = jac[:, :, -1]

        # ode prediction
        state_derivative_analytic = Unicycle.torch_f(
            sampled_input[:, : self.state_dim], sampled_input[:, self.state_dim : -1]
        )

        # error:
        error = state_derivative_numeric - state_derivative_analytic

        # physics cost:
        physics_loss = torch.mean(error**2)
        self.log(
            f"{phase.value}/physics_loss",
            physics_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.data.shape[0],
        )

        return physics_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=trainer_config.lr,
            weight_decay=trainer_config.weight_decay,
        )

    def test_traj(self, test_data: TensorDataset) -> torch.Tensor:
        predictions = self.forward(test_data.data)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        random_idx = np.random.randint(0, test_data.data.shape[0])

        random_traj = test_data.labels[random_idx]
        random_prediction = predictions[random_idx].detach().numpy()

        ax.plot(random_traj[1:, 0], random_traj[1:, 1], "r")
        ax.plot(random_prediction[:-1, 0], random_prediction[:-1, 1], "b")
