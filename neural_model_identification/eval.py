import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from parameters.train_params import TrainParams
from Agent.Imit_Learner import Learner
from data_module.preprocess import DataPreProcess


torch.manual_seed(0)
np.random.seed(0)   

params = TrainParams()
params.models = 'dyn_unicycle'
dataset_tensor, features = DataPreProcess(params).run()

learner = Learner(params, dataset_tensor, use_pretrain=True)

traj = features['eval traj']
print("TRAJ:        ", traj.size())
with torch.no_grad():
    simulated_traj = learner.simulate_trajectory(traj)
    simulated_traj = simulated_traj.cpu().numpy()

traj = traj.cpu().numpy()
plt.plot(simulated_traj[:, 0], simulated_traj[:, 1], label='simulated')
plt.plot(traj[:, 0], traj[:, 1], label='gt')
plt.legend()
plt.show()