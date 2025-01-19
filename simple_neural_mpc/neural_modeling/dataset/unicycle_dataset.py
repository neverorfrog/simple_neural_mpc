import numpy as np
import torch
from scipy.integrate import solve_ivp

from simple_neural_mpc.config.neural_config import DatasetConfig as config
from simple_neural_mpc.neural_modeling.dataset.tensor_dataset import (
    TensorDataset,
)
from simple_neural_mpc.robots.robot import Robot


class UnicycleDataset:

    @staticmethod
    def generate_data(robot: Robot) -> TensorDataset:
        """
        Generates data
        """
        input_dataset = UnicycleDataset.generate_input_data()
        time_range = [0.0, config.len_traj * config.delta_t_for_step]
        time_points = np.arange(
            time_range[0],
            time_range[1] + config.delta_t_for_step,
            config.delta_t_for_step,
        )

        k = config.n_step_constant_input
        X, Y = [], []
        for traj in range(len(input_dataset)):
            traj_input = input_dataset[traj, :, :]

            # integrate the trajectory
            initial_state = np.random.randn(3)
            states = [initial_state]
            controls = []
            times = [np.array([0.0])]

            for i in range(0, len(time_points) - 1):
                t_span = [time_points[i], time_points[i + 1]]
                control = traj_input[i, :]
                sol = solve_ivp(
                    robot.f_expl,
                    t_span,
                    states[-1],
                    args=(control,),
                    t_eval=[t_span[1]],
                    method="RK45",
                )

                states.append(sol.y.squeeze())
                times.append(sol.t)
                controls.append(control)

            s = np.stack(states)
            c = np.stack(controls)
            t = np.stack(times)

            # create sample for dataset
            X_traj_i, Y_traj_i = [], []
            for i in range(0, len(c), k):
                x0 = s[i][np.newaxis, :].repeat(k, axis=0)
                c_substep = c[i : i + k]
                t = np.arange(0, config.delta_t_for_step * k, config.delta_t_for_step)[
                    :, np.newaxis
                ]

                x = np.hstack([x0, c_substep, t])
                X_traj_i.append(x)

                y = s[i : i + k]
                Y_traj_i.append(y)

            X.append(np.concatenate(X_traj_i))
            Y.append(np.concatenate(Y_traj_i))

        X = torch.from_numpy(
            np.stack(X).reshape(-1, config.N_sample, 6), dtype=torch.float32
        )
        Y = torch.from_numpy(
            np.stack(Y).reshape(-1, config.N_sample, 3), dtype=torch.float32
        )
        data = TensorDataset(X, Y)
        return data

    @staticmethod
    def generate_input_data() -> np.ndarray:
        """
        Generates input data for the robot
        """
        trajectory_input = []

        # define range of controls:  +-1 m/s for v and +-1 rad/s for w
        input_range_pos = np.array([-0.25, +1])
        input_range_neg = np.array([-1, +0.25])

        # 1 straight line forward -> [v > 0, w = 0]
        v = np.random.uniform(*input_range_pos, (config.N_sample, config.len_traj, 1))
        w = np.zeros((config.N_sample, config.len_traj, 1))
        trajectory_input.append(np.concatenate([v, w], axis=-1))

        # 2 straight line backward -> [v < 0, w = 0]
        v = np.random.uniform(*input_range_neg, (config.N_sample, config.len_traj, 1))
        w = np.zeros((config.N_sample, config.len_traj, 1))
        trajectory_input.append(np.concatenate([v, w], axis=-1))

        # 3 arc of circle (left) forward ->  [v > 0, w > 0]
        v = np.random.uniform(*input_range_pos, (config.N_sample, config.len_traj, 1))
        w = np.random.uniform(*input_range_pos, (config.N_sample, config.len_traj, 1))
        trajectory_input.append(np.concatenate([v, w], axis=-1))

        # 4 arc of circle (left) backward -> [v < 0, w > 0]
        v = np.random.uniform(*input_range_neg, (config.N_sample, config.len_traj, 1))
        w = np.random.uniform(*input_range_pos, (config.N_sample, config.len_traj, 1))
        trajectory_input.append(np.concatenate([v, w], axis=-1))

        # 5 arc of circle (right) forward -> [v > 0, w < 0]
        v = np.random.uniform(*input_range_pos, (config.N_sample, config.len_traj, 1))
        w = np.random.uniform(*input_range_neg, (config.N_sample, config.len_traj, 1))
        trajectory_input.append(np.concatenate([v, w], axis=-1))

        # 6 arc of circle (right) backward ->  [v < 0, w < 0]
        v = np.random.uniform(*input_range_neg, (config.N_sample, config.len_traj, 1))
        w = np.random.uniform(*input_range_neg, (config.N_sample, config.len_traj, 1))
        trajectory_input.append(np.concatenate([v, w], axis=-1))

        # 7 pure rotation (right) -> [v = 0, w > 0]
        v = np.zeros((config.N_sample, config.len_traj, 1))
        w = np.random.uniform(*input_range_pos, (config.N_sample, config.len_traj, 1))
        trajectory_input.append(np.concatenate([v, w], axis=-1))

        # 8 pure rotation (left) -> [v = 0, w < 0]
        v = np.zeros((config.N_sample, config.len_traj, 1))
        w = np.random.uniform(*input_range_neg, (config.N_sample, config.len_traj, 1))
        trajectory_input.append(np.concatenate([v, w], axis=-1))

        input_dataset = np.concatenate(trajectory_input, axis=0)
        for i in range(0, input_dataset.shape[1], config.n_step_constant_input):
            input_dataset[:, i : i + config.n_step_constant_input, :] = input_dataset[
                :, i, :
            ][:, np.newaxis, :]

        return input_dataset
