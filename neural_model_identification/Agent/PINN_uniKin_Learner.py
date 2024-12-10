from Agent.AbstractLearner import AbstractLearner
from Agent.AbstractLearner import AbstractLearner
from Agent.neural_nets.MLP import MLP
from Agent.dyn_utils.utils import euler_integration
import torch 




class PINNLearner(AbstractLearner):

    def __init__(self, params, demos):
        
        super().__init__()

        self.batch_size = params.batch_size
        self.device = params.device

        self.demos = demos.to(self.device) # in the form of [idx, horizon, (x, u, t)]
        self.n_demos = len(demos)
        self.state_dim = params.state_dim
        self.action_dim = params.input_dim
        self.horizon = params.horizon


        self.model = MLP(
            state_dim=params.state_dim,
            input_dim=params.input_dim,
            n_hidden_layer=params.n_hidden_layer,
            latent_dim=params.latent_dim, 
            time=True
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay
            )

        # loss computing stuff
        # --------------------

        self.jac_funct = torch.func.jacrev(self.model)
        self.mse = torch.nn.MSELoss()
        

    def propagate(self, input_tensor):
        # no euler integration in this learner 
        x_next = self.model(input_tensor)
        return x_next

    def calc_loss(self, x):

        self.imit_weight = 1.0
        loss_imit = self.imit_weight * self.imit_loss(x)
        
        # self.physic_weight = 1.0
        # loss_physic = self.physic_weight * self.physic_loss()

        # self.boundary_weight = 0.8 
        # loss_boundary = self.boundary_weight * self.boundary_loss()

        # return loss_imit + loss_physic + loss_boundary
        return loss_imit
        
    def boundary_loss(self):

        # zero velocity = no motion 
        sampled_zero_input = torch.randn(32, 6)
        sampled_zero_input[:, self.state_dim:-1] = 0
        sampled_zero_input.requires_grad = True
        sampled_zero_input = sampled_zero_input.to(self.device)

        jac = torch.vmap(self.jac_funct)(sampled_zero_input)
        dx_dt = jac[:, :, -1]
        ode_model = torch.zeros_like(dx_dt)

        cost = torch.mean((ode_model - dx_dt)**2)
        return cost
        
    def physic_loss(self):

        sampled_points = torch.randn(32, 6)
        sampled_points.requires_grad = True
        sampled_points = sampled_points.to(self.device)

        jac = torch.vmap(self.jac_funct)(sampled_points)
        dx_dt = jac[:, :, -1]

        state_sampled = sampled_points[:, :self.state_dim]
        action_sampled = sampled_points[:, self.state_dim:-1]

        # forward pass of kin models
        ode_model = torch.empty(32, 3).to(self.device)
        ode_model[:, 0] = torch.cos(state_sampled[:, -1]) * action_sampled[:, 0]
        ode_model[:, 1] = torch.sin(state_sampled[:, -1]) * action_sampled[:, 0]
        ode_model[:, 2] = action_sampled[:, 1]
        
        cost = torch.mean((ode_model - dx_dt)**2)

        return cost 


    def imit_loss(self, x):

        states = x[:, :, :self.state_dim]   # [batch, horizon, 3]
        actions = x[:, :, self.state_dim:-1]  # [batch, horizon, 2]
        t = x[:, :, -1].unsqueeze(-1)  # [batch, horizon]

        next_pair = [torch.hstack((
            states[:, 0, :],
            actions[:, 0, :],
            t[:, 0, :]
            ))]

        next_states = [states[:, 0, :].unsqueeze(-1)]
                
        for k in range(self.horizon-1):

            x_t_next = self.propagate(next_pair[-1])

            next_states.append(x_t_next.unsqueeze(-1))
            next_pair.append(torch.hstack((
                x_t_next,
                actions[:, k+1, :],
                t[:, k+1, :]
            )))

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



