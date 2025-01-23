from dataclasses import dataclass

from simple_neural_mpc.utils import project_root


@dataclass
class DatasetConfig:
    name: str = "trajectories"
    load_data: bool = False
    batch_size: int = 1
    N_type: int = 9  # number of types of trajectories
    N_sample: int = (
        50  # number of trajectories for type (full dataset will be [N_sample * N_type, len_traj, 2])
    )
    len_traj: int = 100  # length of trajectory (number of steps)
    n_step_constant_input: int = 10  # number of steps with constant input
    delta_t_for_step: float = 0.01
    ratios = [0.8, 0.15, 0.05]


@dataclass
class TrainerConfig:
    wandb_api_key: str = "41e4ba7425e35355cd4456863ed4cd9c73c084a3"
    wandb_project: str = "neural_mpc_mlp"
    ckpt_path: str = f"{project_root()}/ckpt"
    patience: int = 10
    min_delta: float = 0.0001
    max_epochs: int = 50
    resume_training: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-8


@dataclass
class PinnConfig:
    latent_dim: int = 128
    imit_loss_weight: float = 1.0
    physics_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    particle_batch_size_boundary: int = 1024
    particle_batch_size_gradient: int = 1024
