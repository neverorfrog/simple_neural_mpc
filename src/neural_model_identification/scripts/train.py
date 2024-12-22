import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from neural_model_identification.parameters.train_params import TrainParams

from neural_model_identification.learner.mlp_learner import Learner

from neural_model_identification.data_generation.preprocess import DataPreProcess

torch.manual_seed(0)
np.random.seed(0)   

dataset_tensor, features = DataPreProcess().run()
learner = Learner(dataset_tensor)

with tqdm(total = TrainParams.train_step) as pbar:
    for i in range(TrainParams.train_step):
      loss = learner.train_step()
      if i % 100 == 0:
        pbar.set_description(f"LOSS: {loss:.2f}")

    #@ TODO : implement eval
    # if i % params.eval_step == 0:
        
        # eval
        # save_weights
        # save_metrics
        # generate some outputs

print("Training ended .............")
learner.save()
# # eval over traj_raw:
# # -------------------
# traj = features['eval traj']
# # traj = features['trajectory raws'][0]
# print("TRAJ:        ", traj.size())
# # T = torch.arange(0, 0.3*traj.shape[0], 0.3)
# # traj = torch.cat((traj, T.unsqueeze(-1)), dim=-1)
# with torch.no_grad():
#     simulated_traj = learner.simulate_trajectory(traj)
#     simulated_traj = simulated_traj.cpu().numpy()


# traj = traj.cpu().numpy()
# plt.plot(simulated_traj[:, 0], simulated_traj[:, 1], label='simulated')
# plt.plot(traj[:, 0], traj[:, 1], label='gt')
# plt.legend()
# plt.show()