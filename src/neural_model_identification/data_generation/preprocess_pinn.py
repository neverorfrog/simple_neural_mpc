import os, sys
sys.path.append('.')

import numpy as np
import torch


from neural_model_identification.parameters.train_params_pinn import TrainParamsPinn

class DataPreProcess:

    def __init__(self):

        self.dataset = None

        self.normalize = TrainParamsPinn.normalize_data
        self.add_noise_in_reading = TrainParamsPinn.add_noise_in_reading
        self.data_path = TrainParamsPinn.data_path
        self.delta_t = TrainParamsPinn.dt
        self.n_step = TrainParamsPinn.n_step
        self.type_data = TrainParamsPinn.models


    def run(self):

        full_dataset = []
        full_traj_raw = []
        eval_traj = []

        for folder in os.listdir(self.data_path):

            path = os.path.join(self.data_path, folder)
            try:
                traj = self.read_files(path)
            except NotADirectoryError:
                continue

            #trajectory_raw = np.concatenate([states_raw, actions_raw], axis=1)
            if folder.endswith("0"):
                eval_traj.append(traj)
                continue
            
            full_dataset.append(traj)

        self.dataset = torch.FloatTensor(np.stack(full_dataset)).reshape(-1, 9)
        eval_traj = torch.FloatTensor(np.stack(eval_traj)).reshape(-1, 9)
        extra_features = {
            'trajectory raws': None, 'eval traj': eval_traj
        }

        return self.dataset, extra_features
    

    def read_files(self, path):

        states = np.load(os.path.join(path, "states.npy"))
        actions = np.load(os.path.join(path, "actions.npy"))

        x0 = states[0]
        x0 = x0[:, np.newaxis].repeat(self.n_step, axis=1).T
        time = np.arange(0, self.n_step*self.delta_t, self.delta_t)
        X = np.hstack([x0, actions, time[:, np.newaxis]])
        Y = states[:-1]
        return np.hstack([X, Y])
