import torch
import torch.nn as nn

from .highway import HighwayLayer
from .integration import IntegrationLayer

class Sine(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

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
        state_dim,
        input_dim,
        latent_dim,
        is_highway=False,
        is_residual=False,
        is_in_mpc=False,
    ) -> None:
        super(MLP, self).__init__()

        self.input_shape = state_dim + input_dim
        self.state_dim = state_dim

        self.is_in_mpc = is_in_mpc
        self.is_highway = is_highway
        self.is_residual = is_residual
        self.highway = HighwayLayer(state_dim)
        self.integration = IntegrationLayer(state_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.input_shape, latent_dim),
            Sine(),
            nn.Linear(latent_dim, latent_dim),
            Sine(),
            nn.Linear(latent_dim, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has dimension [batch_size, state_dim + action_dim + state_dim]
        """
        if self.is_in_mpc:
            input = x[:, : self.input_shape].T
        else:
            input = x[:, : self.input_shape]

        fc_output = self.fc(input)

        if self.is_residual:
            if self.is_in_mpc:
                derivative = x[:, self.input_shape : self.input_shape + self.state_dim].T
            else:
                derivative = x[:, self.input_shape : self.input_shape + self.state_dim]
            fc_output = self.highway(derivative, fc_output)
        elif self.is_highway:
            fc_output = self.integration(input[:, : self.state_dim], fc_output)

        if self.is_in_mpc:
            return fc_output.T
        else:
            return fc_output


class MLP_Pinn(nn.Module):
    """
    this is assumed to simulate the dyn as : \dot x = A x + B u
                                                    ~ [A|B]_\theta [x|u].T
                                                    ~ f_\theta([x|u])

    ps: use tanh --> relu kill negative val and is not suitable for
        dyn sys identification
    """

    def __init__(self, state_dim, input_dim, latent_dim) -> None:
        super(MLP_Pinn, self).__init__()

        self.input_shape = state_dim + input_dim + 1 
        self.state_dim = state_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_shape, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has dimension [batch_size, state_dim + action_dim + 1]
        """
        input = x[:, : self.input_shape].T
        fc_output = self.fc(input)

        return fc_output.T
