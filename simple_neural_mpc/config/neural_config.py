from dataclasses import dataclass

from simple_neural_mpc.utils import project_root


@dataclass
class DatasetConfig:
    name: str = "state"
    load_data: bool = True
    batch_size: int = 32
    N_traj: int = 100_000  # number of trajectories in the dataset
    len_traj: int = 20  # length of trajectory (number of steps)
    n_step_constant_input: int = 5  # number of steps with constant input
    delta_t_for_step: float = 0.005
    upper_bound: float = 5
    lower_bound: float = -5
    ratios = [0.7, 0.25, 0.05]


@dataclass
class TrainerConfig:
    wandb_api_key: str = "41e4ba7425e35355cd4456863ed4cd9c73c084a3"
    wandb_project: str = "simple_neural_mpc"
    ckpt_path: str = f"{project_root()}/simple_neural_mpc/neural_modeling/models"
    patience: int = 15
    min_delta: float = 1e-6
    max_epochs: int = 150
    resume_training: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-8


@dataclass
class PinnConfig:
    latent_dim: int = 128
