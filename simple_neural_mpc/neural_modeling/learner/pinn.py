from enum import Enum

import lightning as L
import torch
from torch.func import jacrev, vmap

from simple_neural_mpc.config.neural_config import PinnConfig as config
from simple_neural_mpc.config.neural_config import (
    TrainerConfig as trainer_config,
)
from simple_neural_mpc.robots.robot import Robot


class Phase(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Pinn(L.LightningModule):

    def __init__(self, robot: Robot, data_range: torch.Tensor):
        super(Pinn, self).__init__()
        self.robot = robot
        self.data_range = data_range

        self.state_dim = len(self.robot.state)
        self.input_dim = len(self.robot.input)
        nn_input_dim = self.state_dim + self.input_dim + 1
        nn_output_dim = self.state_dim

        self.lower_bound = data_range[0]
        self.upper_bound = data_range[1]

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(nn_input_dim, config.latent_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(config.latent_dim, config.latent_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(config.latent_dim, nn_output_dim),
        )

        self.mse = torch.nn.MSELoss()
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return self.mlp(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad()
        X, Y = batch
        X = X.to(self.device).float()
        Y = Y.to(self.device).float()
        loss = self.compute_loss(X, Y, Phase.TRAIN)
        self.manual_backward(loss)
        opt.step()
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        X, Y = batch
        X = X.to(self.device).float()
        Y = Y.to(self.device).float()
        loss = self.compute_loss(X, Y, Phase.VAL)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        X, Y = batch
        X = X.to(self.device).float()
        Y = Y.to(self.device).float()
        loss = self.compute_loss(X, Y, Phase.TEST)
        return loss

    def compute_loss(self, X: torch.Tensor, Y: torch.Tensor, phase: Phase):
        """
        Computes the loss of the batch
        """
        imit_loss = self.compute_mse_loss(X, Y, phase)
        physics_loss = self.compute_physics_loss(phase)
        boundary_loss = self.compute_boundary_loss(phase)
        total_loss = imit_loss + physics_loss + boundary_loss
        self.log(f"{phase.value}/loss", total_loss)
        return total_loss

    def compute_mse_loss(self, X: torch.Tensor, Y: torch.Tensor, phase: Phase):
        """
        Computes the mean squared error loss
        """
        pred_y = self.forward(X)
        imit_loss = self.mse(pred_y, Y)

        self.log(f"{phase.value}/imit_loss", imit_loss)

        return config.imit_loss_weight * imit_loss

    def compute_physics_loss(self, phase: Phase):
        """
        Computes the physics loss
        """
        N_particle = config.particle_batch_size_gradient

        # sample the particle
        sampled_input = torch.rand(N_particle, self.state_dim + self.input_dim + 1).to(
            self.device
        )
        # normalize the particle in the respective range

        sampled_input = (
            sampled_input * (self.upper_bound - self.lower_bound) + self.lower_bound
        ).float()
        sampled_input.requires_grad = True

        # jacobian --> [batch, state_dim, (state_dim + input_dim + 1(time))]
        jac = vmap(jacrev(self.mlp))(sampled_input).squeeze()
        state_derivative_numeric = jac[:, :, -1]

        # ode prediction
        state_derivative_analytic = self.robot.torch_f(
            sampled_input[:, : self.state_dim], sampled_input[:, self.state_dim : -1]
        )

        # error:
        error = state_derivative_numeric - state_derivative_analytic

        # physics cost:
        physics_loss = torch.mean(error**2)

        self.log(f"{phase.value}/physics_loss", physics_loss)

        return config.physics_loss_weight * physics_loss

    def compute_boundary_loss(self, phase: Phase):
        """
        Computes the boundary loss
        """
        N_particle = config.particle_batch_size_boundary

        # sample the particle
        sampled_input = torch.rand(N_particle, self.state_dim + self.input_dim + 1).to(
            self.device
        )

        # normalize in the respective range
        sampled_input = (
            sampled_input * (self.upper_bound - self.lower_bound) + self.lower_bound
        ).float()

        # zeros the velocity
        sampled_input[:, self.state_dim : self.state_dim + self.input_dim] = 0.0

        # 0 velocity means that we want to predict the initial state
        # the models must be a kind of identity function
        target_prediction = sampled_input[:, : self.state_dim].clone()

        sampled_input.requires_grad = True

        pred_output = self.forward(sampled_input)
        boundary_loss = self.mse(pred_output, target_prediction)

        self.log(f"{phase.value}/boundary_loss", boundary_loss)

        return config.boundary_loss_weight * boundary_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=trainer_config.lr,
            weight_decay=trainer_config.weight_decay,
        )
