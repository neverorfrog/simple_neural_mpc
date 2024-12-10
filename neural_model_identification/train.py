import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from parameters.train_params import TrainParams

from Agent.Imit_Learner import Learner
from Agent.PINN_uniKin_Learner import PINNLearner

from data_module.preprocess import DataPreProcess

torch.manual_seed(0)
np.random.seed(0)   

params = TrainParams()
params.models = 'pinn-like'
dataset_tensor, features = DataPreProcess(params).run()

learner = PINNLearner(params, dataset_tensor)


for i in tqdm(range(params.train_step)):
    learner.train_step()

    #@ TODO : implement eval
    # if i % params.eval_step == 0:
        
        # eval
        # save_weights
        # save_metrics
        # generate some outputs


# eval over traj_raw:
# -------------------
traj = features['eval traj']
traj = features['trajectory raws'][0]
T = torch.arange(0, 0.3*traj.shape[0], 0.3)
traj = torch.cat((traj, T.unsqueeze(-1)), dim=-1)
with torch.no_grad():
    simulated_traj = learner.simulate_trajectory(traj)
    simulated_traj = simulated_traj.cpu().numpy()


traj = traj.cpu().numpy()
plt.plot(simulated_traj[:, 0], simulated_traj[:, 1], label='simulated')
plt.plot(traj[:, 0], traj[:, 1], label='gt')
plt.legend()
plt.show()