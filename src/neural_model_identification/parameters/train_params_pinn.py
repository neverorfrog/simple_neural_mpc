from dataclasses import dataclass

from neural_model_identification.utils.utils import project_root


@dataclass
class TrainParamsPinn:

    # data generations parameters
    # ---------------------------
    n_step = 30 # aka number of action
    dt: float = 0.01  # time step for the euler integration
    n_traj = 5_000  # number of trajectories
    plot_traj = False
    
    is_pinn: bool = False
    models: str = "kin_unicycle"  # --> specify which model use for generate train data
    data_path: str = (
        f"{project_root()}/src/neural_model_identification/data/{models}"  # path to the folder containing the data_file
    )
    model_path: str = (
        f"{project_root()}/src/neural_model_identification/trained_models/{models}/model.pth"
    )
    dynamical_order: int = 1  # --> dynamical order of the model

    state_dim: int = 3  # dimension of the state x
    input_dim: int = 2  # dimension of the input u

    latent_dim: int = 128  # dimension of the latent space
    n_hidden_layer = 4 # number of hidden layers


    batch_size: int = 1024  
    particle_batch_size_boundary = 2000
    particle_batch_size_gradient = 2000

    normalize_data: bool = False
    add_noise_in_reading: bool = False

    # neural model parameters
    # -----------------------
    neural_net: str = "mlp"  # soon : bbn, lnn
    add_noise_in_train: bool = (
        False  # corrupt the input of the models [just for robustify]
    )
    loss: str = (
        "mse"  # other option: soft-DTWD loss : https://tslearn.readthedocs.io/en/stable/auto_examples/autodiff/plot_soft_dtw_loss_for_pytorch_nn.html
    )
    # for PINN: use lagrangian loss etc @TODO
    lr: float = 1e-6  # ADAM learning rate
    weight_decay: float = 1e-9  # ADAM weight decay

    imit_loss_weight = 1.0
    boundary_loss_weight = 0.1
    physic_loss_weight = 0.1

    #train_step: int = 10000  # number of training steps

    device: str = "cpu"

    # dynamics parameters
    # -------------------
    integration_method: str = "euler"  # for now only euler
    const_delta_t: bool = (
        True  # specify if the delta t in the ode solver is constant or considered as part of the data
    )
    # if False, use the delta_t_val define below
