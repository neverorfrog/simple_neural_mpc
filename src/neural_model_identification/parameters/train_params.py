from dataclasses import dataclass

from neural_model_identification.utils.utils import project_root


@dataclass
class TrainParams:

    # data generations parameters
    # ---------------------------
    n_step = 1000  # aka number of action
    n_traj = 200  # number of trajectories
    plot_traj = False

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

    latent_dim: int = 256  # dimension of the latent space

    # batch_size must be such that the following expression returns an integer number (not a float)
    # otherwise the last batch for the training has different dimensions and it does not work.
    # Expression:           ((n_steps - horizon) * n_traj) / batch_size
    horizon: int = 20  # prediction horizion for multi-step roll-out
    batch_size: int = 16  # you know this

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
    lr: float = 1e-4  # ADAM learning rate
    weight_decay: float = 1e-5  # ADAM weight decay
    train_step: int = 10000  # number of training steps

    device: str = "cpu"

    # dynamics parameters
    # -------------------
    integration_method: str = "euler"  # for now only euler
    const_delta_t: bool = (
        True  # specify if the delta t in the ode solver is constant or considered as part of the data
    )
    # if False, use the delta_t_val define below
    dt: float = 0.01  # time step for the euler integration
