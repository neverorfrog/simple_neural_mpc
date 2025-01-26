import torch
import torch.nn as nn

from simple_neural_mpc.config.neural_config import PinnConfig as config


class MLP(nn.Module):
    """
    this is assumed to simulate the dyn as : \dot x = A x + B u
                                                    ~ [A|B]_\theta [x|u].T
                                                    ~ f_\theta([x|u])

    ps: use tanh --> relu kill negative val and is not suitable for
        dyn sys identification
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        activation: nn.Module,
        predicts_state: bool = True,
        in_mpc: bool = False,
        is_highway: bool = False,
    ) -> None:
        super(MLP, self).__init__()
        self.in_mpc = in_mpc
        self.is_highway = is_highway
        self.nn_input_dim = state_dim + input_dim + 1 if predicts_state else state_dim + input_dim
        self.nn_output_dim = state_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.nn_input_dim, config.latent_dim),
            activation,
            torch.nn.Linear(config.latent_dim, config.latent_dim),
            activation,
            torch.nn.Linear(config.latent_dim, self.nn_output_dim),
        )

        if self.is_highway:
            self.skip = nn.Linear(state_dim, state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has dimension [batch_size, traj_len, state_dim + action_dim + 1]
        """
        if x.ndim == 2 and not self.in_mpc:
            x = x.unsqueeze(0)

        if self.in_mpc:
            input = x.T
        else:
            input = x
        fc_output = self.fc(input)  # [batch_size, traj_len, state_dim]

        if self.is_highway:
            if self.in_mpc:
                skip_output = self.skip(input.T[: self.nn_output_dim].T)
            else:
                skip_output = self.skip(input[:, :, : self.nn_output_dim])
            fc_output += skip_output

        output = fc_output.T if self.in_mpc else fc_output
        return output


class Sine(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)
