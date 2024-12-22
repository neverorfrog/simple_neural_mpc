import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    this is assumed to simulate the dyn as : \dot x = A x + B u
                                                    ~ [A|B]_\theta [x|u].T
                                                    ~ f_\theta([x|u])

    ps: use tanh --> relu kill negative val and is not suitable for
        dyn sys identification
    """

    def __init__(self, state_dim, input_dim, n_hidden_layer, latent_dim):
        super(MLP, self).__init__()

        input_shape = state_dim + input_dim

        self.fc_input = nn.Linear(input_shape, latent_dim)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(n_hidden_layer)]
        )
        self.fc_output = nn.Linear(latent_dim, state_dim)

    def forward(self, x):

        # x is assumed to be [x|u] concatenated
        x = torch.tanh(self.fc_input(x))
        for fc in self.fc_hidden:
            x = torch.tanh(fc(x))
        x = self.fc_output(x)
        return x
