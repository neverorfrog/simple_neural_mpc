import torch
import torch.nn as nn

from simple_neural_mpc.robots.robot import Robot
from simple_neural_mpc.config.neural_config import PinnConfig as config

class MLP(nn.Module):
    """
    this is assumed to simulate the dyn as : \dot x = A x + B u
                                                    ~ [A|B]_\theta [x|u].T
                                                    ~ f_\theta([x|u])

    ps: use tanh --> relu kill negative val and is not suitable for
        dyn sys identification
    """

    def __init__(self, state_dim: int, input_dim: int, in_mpc: bool = True) -> None:
        super(MLP, self).__init__()
        self.in_mpc = in_mpc
        self.nn_input_dim = state_dim + input_dim + 1
        self.nn_output_dim = state_dim
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.nn_input_dim, config.latent_dim),
            Sine(),
            torch.nn.Linear(config.latent_dim, config.latent_dim),
            Sine(),
            torch.nn.Linear(config.latent_dim, config.latent_dim),
            Sine(),
            torch.nn.Linear(config.latent_dim, self.nn_output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has dimension [batch_size, state_dim + action_dim + 1]
        """
        if self.in_mpc:
            input = x[:, : self.nn_input_dim].T
        else:
            input = x
        fc_output = self.mlp(input)
        output = fc_output.T if self.in_mpc else fc_output
        return output
    
    
class Sine(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)