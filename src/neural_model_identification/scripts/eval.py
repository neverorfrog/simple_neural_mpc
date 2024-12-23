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
        plt.plot(simulated_traj[:, 0], simulated_traj[:, 1], label="simulated")
        plt.plot(traj_array[:, 0], traj_array[:, 1], label="gt")
        plt.legend()
        plt.show()
