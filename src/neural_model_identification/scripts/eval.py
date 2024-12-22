import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_model_identification.data_generation.preprocess import (
    DataPreProcess,
)
from neural_model_identification.learner.mlp_learner import Learner

torch.manual_seed(0)
np.random.seed(0)
dataset_tensor, features = DataPreProcess().run()
learner = Learner(dataset_tensor, use_pretrain=True)

traj = features["eval traj"]
print("TRAJ:        ", traj.size())
with torch.no_grad():
    simulated_traj = learner.simulate_trajectory(traj)
    simulated_traj = simulated_traj.cpu().numpy()

traj = traj.cpu().numpy()
plt.plot(simulated_traj[:, 0], simulated_traj[:, 1], label="simulated")
plt.plot(traj[:, 0], traj[:, 1], label="gt")
plt.legend()
plt.show()
