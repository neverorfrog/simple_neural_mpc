import os

import torch

from neural_model_identification.learner.abstract_learner import AbstractLearner
from neural_model_identification.learner.nn.mlp import MLP
from neural_model_identification.parameters.train_params import TrainParams
from neural_model_identification.utils.utils import euler_integration


class Learner(AbstractLearner):

    def __init__(self, use_pretrain=False):

        super().__init__()
        self.model_path = TrainParams.model_path

        self.batch_size = TrainParams.batch_size
        self.device = TrainParams.device

        self.state_dim = TrainParams.state_dim
        self.input_dim = TrainParams.input_dim
        self.horizon = TrainParams.horizon

        self.model = MLP(
            state_dim=TrainParams.state_dim,
            input_dim=TrainParams.input_dim,
            latent_dim=TrainParams.latent_dim,
        ).to(self.device)

        if use_pretrain:
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=TrainParams.lr,
            weight_decay=TrainParams.weight_decay,
        )

        self.mse = torch.nn.MSELoss(reduction="sum")

    def propagate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        input: tensor of [batch, state_dim + input_dim + state_dim]
        predict next x_dot and integrate in the models
        """
        
        if TrainParams.is_pinn is True:
            return self.model(input_tensor)
        else:
            x_t_dot = self.model(input_tensor)
            return euler_integration(
                input_tensor[:, : self.state_dim], x_t_dot, delta_t=TrainParams.dt
            )

    def calc_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        here we propagate the sample through the model and calculate the loss
        """
        states = x[:, :, : self.state_dim]  # [batch, horizon, state_dim]
        actions = x[
            :, :, self.state_dim : self.state_dim + self.input_dim
        ]  # [batch, horizon, input_dim]

        next_pair = [
            torch.hstack((states[:, 0, :], actions[:, 0, :]))
        ]  # [batch, state_dim + input_dim]
        next_states = [states[:, 0, :].unsqueeze(-1)]  # [batch, state_dim , 1]

        for k in range(self.horizon - 1):
            x_t_next = self.propagate(next_pair[-1])  # [batch, state_dim]
            next_states.append(x_t_next.unsqueeze(-1))
            next_pair.append(torch.hstack((x_t_next, actions[:, k + 1, :])))

        generated_traj = torch.concatenate(
            next_states, dim=-1
        )  # [batch, state_dim, horizon]
        generated_traj = generated_traj.permute(0, 2, 1)  # [batch, horizon, state_dim]

        loss = self.mse(states, generated_traj)

        return loss

    def simulate_trajectory(self, raw_trajectory):
        """do a roll-out along the trajectory input"""
        raw_trajectory = torch.FloatTensor(raw_trajectory).to(self.device)
        x = raw_trajectory[0, : self.state_dim]
        states = [x]

        # the net require input of shape (batch=1, state_dim + input_dim)
        # add the batch dim
        x = x.unsqueeze(0)

        for i in range(len(raw_trajectory) - 1):

            # create the input tensor as concatenation of state and action
            input_tensor = torch.hstack(
                (x, raw_trajectory[i, self.state_dim :].unsqueeze(0))
            )
            x = self.propagate(input_tensor)
            states.append(x.squeeze(0))

        return torch.stack(states, dim=0)

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
