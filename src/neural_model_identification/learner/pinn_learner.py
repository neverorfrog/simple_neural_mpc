import torch
import torch.nn 

from torch.func import (vmap, jacrev)


class MLP(torch.nn.Module):

    def __init__(
        self,
        state_dim,
        input_dim,
        latent_dim,
        n_hidden_layer=2,
        time=False
    ):
        super(MLP, self).__init__()

        self.input_shape = state_dim + input_dim
        self.state_dim = state_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape, latent_dim),
            torch.nn.Sine(),
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.Sine(),
            torch.nn.Linear(latent_dim, state_dim),
        )

        self.time = time
        if self.time:
            self.fc.add_module('time', torch.nn.Linear(1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has dimension [batch_size, state_dim + action_dim + state_dim]
        """
        return self.fc(x)


class Pinn_learner:


    def __init__(
        self,
        ode_fn,
        params,
        datset,
        input_range,
        use_pretrain=False, 
        device="cuda"):
        
        
        # robot model, dynamics
        self.state_dim = params.state_dim
        self.input_dim = params.input_dim
        self.ode_fn = ode_fn


        self.model_path = params.model_path
        self.device = device
        self.model = MLP(
            state_dim=params.state_dim,
            input_dim=params.input_dim,
            n_hidden_layer=params.n_hidden_layer,
            latent_dim=params.latent_dim,
            time=True
        ).to(self.device)

        if params.use_pretrain:
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        
        # dataset 
        self.dataset = datset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=params.batch_size, 
            shuffle=True
        )

        self.particle_batch_size_boundary = 100
        self.particle_batch_size_gradient = 100

        self.max_range = input_range[0].to(self.device)
        self.min_range = input_range[1].to(self.device) 
        #self.batch_size = None
        #self.particle_batch_size_boundary = None
        #self.particle_batch_size_gradient = None

        #self.config_batch_size(
        #    params.batch_size, 
        #    params.particle_batch_size_boundary,
        #    params.particle_batch_size_gradient
        #)

        # training component 
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay
        )

        self.imit_loss_weight = params.imit_loss_weight
        self.boundary_loss_weight = params.boundary_loss_weight
        self.physic_loss_weight = params.physic_loss_weight

        self.mse = torch.nn.MSELoss()


    # config functions
    # -------------------
    def config_batch_size(
        self,
        batch_size,
        particle_batch_size_boundary,
        particle_batch_size_gradient):

        if batch_size == -1:
            self.batch_size = len(self.dataset)
        else:
            self.batch_size = batch_size
        
        if particle_batch_size_boundary == -1:
            self.particle_batch_size_boundary = len(self.dataset)//2
        else:
            self.particle_batch_size_boundary = particle_batch_size_boundary

        if particle_batch_size_gradient == -1:
            self.particle_batch_size_gradient = len(self.dataset)//2
        else:
            self.particle_batch_size_gradient = particle_batch_size_gradient


    # training functions
    # -------------------
    def calc_loss(self, batch):
        
        statistics = {}
        if self.imit_loss_weight > 0:
            imit_loss = self.calc_mse_loss(batch)
            statistics['imit_loss'] = imit_loss.item()
        
        if self.boundary_loss_weight > 0:
            boundary_loss = self.calc_boundary_loss()
            statistics['boundary_loss'] = boundary_loss.item()

        if self.physic_loss_weight > 0:
            physic_loss = self.calc_physic_loss()
            statistics['physic_loss'] = physic_loss.item()

        loss = imit_loss + boundary_loss + physic_loss
        statistics['total_loss'] = loss.item()

        return loss, statistics

    def train_step(self):
        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            loss, stat = self.calc_loss(batch)
            loss.backward()
            self.optimizer.step()
        return stat


    # loss definition
    # -------------------
    def calc_physic_loss(self):

        N_particle = self.particle_batch_size_gradient

        # sample the particle        
        sampled_input = torch.rand(N_particle, self.state_dim + self.input_dim + 1).to(self.device)
        # normalize the particle in the respective range
        sampled_input = sampled_input * (self.max_range - self.min_range) + self.min_range
        sampled_input.requires_grad = True

        # pinn prediction
        pred_output = self.model(sampled_input)

        # jacobian --> [batch, state_dim, (state_dim + input_dim + 1(time))]
        jac = vmap(jacrev(self.model))(sampled_input).squeeze()
        state_derivative_numeric = jac[:, :, -1]        

        # ode prediction
        state_derivative_analytic = self.ode_fn(
            sampled_input[:, :self.state_dim],
            sampled_input[:, self.state_dim:-1]
        )

        # error:  
        error = state_derivative_numeric - state_derivative_analytic

        # scale up the error over the orientation:
        error[:, 2] = error[:, 2] * 1.8

        # physics cost:
        physic_loss = torch.mean(error**2)
                        #torch.mean(
            #    torch.square(state_derivative_numeric - state_derivative_analytic),
            #    dim=1
        

        return self.physic_loss_weight * physic_loss

    def calc_mse_loss(self, batch):
        X, Y = batch[:, :-self.state_dim], batch[:, -self.state_dim:]
        pred_y = self.model(X)
        return self.imit_loss_weight * self.mse(pred_y, Y)

    def calc_boundary_loss(self):
        N_particle = self.particle_batch_size_boundary

        # sample the particle
        sampled_input = torch.rand(N_particle, self.state_dim + self.input_dim + 1).to(self.device)
        
        # normalize in the respective range 
        sampled_input = sampled_input * (self.max_range - self.min_range) + self.min_range
        
        # zeros the velocity
        sampled_input[:, self.state_dim:self.state_dim+self.input_dim] = 0.0
        

        # 0 velocity means that we want to predict the initial state 
        # the models must be a kind of identity function 
        target_prediction = sampled_input[:, :self.state_dim].clone()

        sampled_input.requires_grad = True
        pred_output = self.model(sampled_input)
        boundary_loss = self.mse(pred_output, target_prediction)

        return self.boundary_loss_weight * boundary_loss








