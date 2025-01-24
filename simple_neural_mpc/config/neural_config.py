from dataclasses import dataclass

from simple_neural_mpc.utils import project_root


@dataclass
class DatasetConfig:
    name: str = "trajectories"
    load_data: bool = True
    batch_size: int = 64
    N_type: int = 9  # number of types of trajectories
    N_sample: int = (
        50  # number of trajectories for type (full dataset will be [N_sample * N_type, len_traj, 2])
    )
    N_traj: int = 50_000  # number of trajectories in the dataset
    len_traj: int = 20  # length of trajectory (number of steps)
    n_step_constant_input: int = 5  # number of steps with constant input
    delta_t_for_step: float = 0.01
    ratios = [0.7, 0.25, 0.05]


@dataclass
class TrainerConfig:
    wandb_api_key: str = "41e4ba7425e35355cd4456863ed4cd9c73c084a3"
    wandb_project: str = "neural_mpc_mlp"
    ckpt_path: str = f"{project_root()}/simple_neural_mpc/neural_modeling/models"
    patience: int = 10
    min_delta: float = 1e-6
    max_epochs: int = 100
    resume_training: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-8


@dataclass
class PinnConfig:
    latent_dim: int = 124
    imit_loss_weight: float = 1.0
    physics_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    particle_batch_size_boundary: int = 1024
    particle_batch_size_gradient: int = 1024
