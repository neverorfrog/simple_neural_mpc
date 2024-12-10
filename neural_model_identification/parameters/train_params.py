from dataclasses import dataclass

@dataclass
class TrainParams:

    # data generations parameters
    # ---------------------------

    data_path : str = 'neural_model_identification/data_module/just_2_traj' # path to the folder containing the data_file
    models: str = 'unicycle'  # --> specify which model use for generate train data
    dynamical_order : int = 1 # --> dynamical order of the model

    state_dim: int = 3 # dimension of the state x
    input_dim: int = 2 # dimension of the input u

    n_hidden_layer: int = 3 # number of hidden layers
    latent_dim: int = 30 # dimension of the latent space


    horizon: int = 30 # prediction horizion for multi-step roll-out
    batch_size: int = 32 # you know this

    normalize_data: bool = False
    add_noise_in_reading: bool = False

    # neural model parameters
    # -----------------------
    neural_net: str = 'mlp' # soon : bbn, lnn
    add_noise_in_train: bool = True  # corrupt the input of the models [just for robustify]
    loss:  str = 'mse' # other option: soft-DTWD loss : https://tslearn.readthedocs.io/en/stable/auto_examples/autodiff/plot_soft_dtw_loss_for_pytorch_nn.html 
                       # for PINN: use lagrangian loss etc @TODO
    lr : float = 1e-03               # ADAM learning rate
    weight_decay: float = 1e-4      # ADAM weight decay
    train_step: int = 10_000       # number of training steps

    device: str = 'cuda'

    # dynamics parameters
    # -------------------
    integration_method: str = 'euler'  # for now only euler
    const_delta_t : bool = True   # specify if the delta t in the ode solver is constant or considered as part of the data
                                  # if False, use the delta_t_val define below 
    delta_t_val : float = 0.01
     