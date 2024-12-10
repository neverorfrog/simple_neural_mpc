from Agent.AbstractLearner import AbstractLearner
from Agent.neural_nets.MLP import MLP
from Agent.dyn_utils.utils import euler_integration
import torch



class Learner(AbstractLearner):

    def __init__(self, params, demos):
        
        super().__init__()

        self.batch_size = params.batch_size
        self.device = params.device

        self.delta_t = 0.3
        self.demos = demos.to(self.device) # in the form of [idx, (x, u), horizon]
        self.n_demos = len(demos)
        self.state_dim = params.state_dim
        self.horizon = params.horizon


        self.model = MLP(
            state_dim=params.state_dim,
            input_dim=params.input_dim,
            n_hidden_layer=params.n_hidden_layer,
            latent_dim=params.latent_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay
            )

        self.mse = torch.nn.MSELoss()


    def propagate(self, input_tensor):
        """ 
        input: tensor of [batch, state_dim + input_dim]
        predict next x_dot and integrate in the models 
        """
        x_t_dot = self.model(input_tensor)
        next_x = euler_integration(
            input_tensor[:, :self.state_dim],
            x_t_dot, 
            delta_t=self.delta_t
        )
        return next_x


    def calc_loss(self, x):
        ''' 
        here we propagate the sample through the model and calculate the loss
        '''
        states = x[:, :, :self.state_dim]   # [batch, horizon, 3]
        actions = x[:, :, self.state_dim:]  # [batch, horizon, 2]

        next_pair = [torch.hstack((states[:, 0, :], actions[:, 0, :]))]
        next_states = [states[:, 0, :].unsqueeze(-1)]
                
        for k in range(self.horizon-1):

            x_t_next = self.propagate(next_pair[-1])

            next_states.append(x_t_next.unsqueeze(-1))
            next_pair.append(torch.hstack((x_t_next, actions[:, k+1, :])))

        generated_traj = torch.concatenate(next_states, dim=-1)
        generated_traj = generated_traj.permute(0, 2, 1)

        loss = self.mse(states, generated_traj)
        return loss 


    def sample_data(self):

        return self.demos[
            torch.randint(
                low=0,
                high=self.n_demos,
                size=(self.batch_size,)
            )
        ]


    def simulate_trajectory(self, raw_trajectory):

        """do a roll-out along the trajectory input"""
        raw_trajectory = torch.FloatTensor(raw_trajectory).to(self.device)
        x = raw_trajectory[0, :self.state_dim]
        states = [x]

        # the net require input of shape (batch=1, state_dim + input_dim)
        # add the batch dim 
        x = x.unsqueeze(0)

        for i in range(len(raw_trajectory) - 1):

            # create the input tensor as concatenation of state and action
            input_tensor = torch.hstack(
                (x, raw_trajectory[i, self.state_dim:].unsqueeze(0))
            )
            x = self.propagate(
                input_tensor
                )
            states.append(x.squeeze(0))

        return torch.stack(states, dim=0)

