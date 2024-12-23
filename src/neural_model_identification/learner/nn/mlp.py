import torch
import torch.nn as nn

from .highway import HighwayLayer


class MLP(nn.Module):
    """
    this is assumed to simulate the dyn as : \dot x = A x + B u
                                                    ~ [A|B]_\theta [x|u].T
                                                    ~ f_\theta([x|u])

    ps: use tanh --> relu kill negative val and is not suitable for
        dyn sys identification
    """

    def __init__(self, state_dim, input_dim, latent_dim, is_highway=False, is_in_mpc=False):
        super(MLP, self).__init__()

        input_shape = state_dim + input_dim
        
        self.is_in_mpc = is_in_mpc
        self.is_highway = is_highway
        self.highway = HighwayLayer(state_dim)

        self.fc = nn.Sequential(
            nn.Linear(input_shape, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, state_dim),
        )

    def forward(self, x):
        """
        x has dimension [batch_size, state_dim + action_dim + state_dim]
        """
        if self.is_in_mpc:
            input = x[:, :5].T
        else:
            input = x[:, :5]
            
        x = self.fc(input)

        if self.is_highway:
            derivative = x[:, 5:].T
            x = self.highway(derivative, x)

        if self.is_in_mpc:
            return x.T
        else:
            return x
