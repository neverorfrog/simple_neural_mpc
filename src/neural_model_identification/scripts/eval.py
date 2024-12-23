import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_model_identification.data_generation.preprocess import (
    DataPreProcess,
)
from neural_model_identification.learner.mlp_learner import Learner

torch.manual_seed(0xDEADBEEF)
np.random.seed(0xDEADBEEF)

dataset_tensor, features = DataPreProcess().run()
learner = Learner(use_pretrain=True)

trajectories = features["eval traj"]

with torch.no_grad():
    for traj in trajectories:
        simulated_traj = learner.simulate_trajectory(traj)
        simulated_traj = simulated_traj.cpu().numpy()

        traj_array = traj.cpu().numpy()

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        axes: plt.Axes = axs[0]
        axes.plot(simulated_traj[:, -1], simulated_traj[:, 0], label="simulated")
        axes.plot(simulated_traj[:, -1], traj_array[:, 0], label="ground truth")
        axes.set_title("X")
        axes.legend()

        axes: plt.Axes = axs[1]
        axes.plot(simulated_traj[:, -1], simulated_traj[:, 1], label="simulated")
        axes.plot(simulated_traj[:, -1], traj_array[:, 1], label="ground truth")
        axes.set_title("Y")
        axes.legend()

        axes: plt.Axes = axs[2]
        axes.plot(simulated_traj[:, -1], simulated_traj[:, 2], label="simulated")
        axes.plot(simulated_traj[:, -1], traj_array[:, 2], label="ground truth")
        axes.set_title("Theta")
        axes.legend()

        plt.tight_layout()
        plt.show()
