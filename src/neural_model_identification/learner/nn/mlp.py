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

    def __init__(self, state_dim, input_dim, latent_dim):
        super(MLP, self).__init__()

        input_shape = state_dim + input_dim
        
        self.highway = HighwayLayer(state_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(input_shape, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, state_dim),
        )

    def forward(self, x):
        '''
        x has dimension [batch_size, state_dim + action_dim + state_dim]
        '''
        input = x[:, :5]
        fc_output = self.fc(input)
        derivative = x[:, 5:]
        x = self.highway(derivative,fc_output)
        return x
