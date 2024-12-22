import os

import numpy as np
import torch

from neural_model_identification.parameters.train_params import TrainParams


class DataPreProcess:
    # read a set of trajectories from a specified folder
    # generate transition data from each trajectories
    # normalize data if needed and apply any things (noise? augmentation?)
    def __init__(self):
        self.dataset = None

        self.normalize = TrainParams.normalize_data
        self.add_noise_in_reading = TrainParams.add_noise_in_reading
        self.data_path = TrainParams.data_path
        self.horizon = TrainParams.horizon
        self.delta_t = TrainParams.dt
        self.type_data = TrainParams.models

    def run(self):

        full_dataset = []  # composed of transitions
        full_traj_raw = []  # store the original traj

        for folder in os.listdir(self.data_path):

            full_path = os.path.join(self.data_path, folder)
            # read numpy file, both states and actions.
            try:
                trajectory_raw = self.read_files(full_path)
            except NotADirectoryError:
                continue

            if folder.endswith("0"):
                # save the first trajectory of the folder for evaluation
                eval_traj = trajectory_raw
                continue

            # sample some transtitions form the trajectory
            transitions = self.extract_trajectory(trajectory_raw)

            # store in a list for a obtain a bigger dataset
            full_dataset.append(transitions)
            full_traj_raw.append(trajectory_raw)

        self.dataset = torch.concatenate(full_dataset)

        # permute the transitions by row:
        # this stuff is done also by a dataloader but
        # in this way we can use different methods
        self.dataset = self.dataset[torch.randperm(self.dataset.shape[0])]

        # add any stuff for pre_processing here
        extra_features = {"trajectory raws": full_traj_raw, "eval traj": eval_traj}

        return self.dataset, extra_features

    def read_files(self, path) -> torch.Tensor:

        states_raw = np.load(path + "/states.npy")
        actions_raw = np.load(path + "/actions.npy")

        # we discard the last input, we just don't care about it
        traj = np.hstack((states_raw[:-1, :], actions_raw))
        return torch.FloatTensor(traj)

    def extract_trajectory(self, trajectory) -> torch.Tensor:

        time_batch = []
        state_batch = []
        T = 0
        for i in range(trajectory.shape[0] - self.horizon):

            time_batch.append(
                torch.linspace(T, T + (self.horizon) * self.delta_t, self.horizon)
            )
            state_batch.append(trajectory[i : i + self.horizon])

            T += self.delta_t * (self.horizon)

        transitions = torch.stack(state_batch, dim=0)
        return transitions


def test():
    # test the data pre-process
    import os
    import sys

    print(os.getcwd())
    sys.path.append(".")

    from neural_model_identification.parameters.train_params import TrainParams

    params = TrainParams()
    params.data_path = "neural_model_identification/data_module/trajectories_dyn"

    print(os.listdir(params.data_path))

    data, _ = DataPreProcess(params).run()
    print(data.shape)
