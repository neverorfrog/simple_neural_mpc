from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

from simple_neural_mpc.config.neural_config import DatasetConfig as config
from simple_neural_mpc.neural_modeling.dataset.tensor_dataset import (
    TensorDataset,
)
from simple_neural_mpc.robots.unicycle import Unicycle


class UnicycleDataset:

    @staticmethod
    def generate_data(robot: Unicycle) -> TensorDataset:
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
        for j in range(len(input_dataset)):
            traj_input = input_dataset[j, :, :]

            # integrate the trajectory
            initial_state = np.random.uniform(-1, 1, 3)
            initial_state[2] = np.random.uniform(-np.pi, np.pi)
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

            # print(s.shape)
            # print(initial_state[:2])
            # print(np.rad2deg(initial_state[2]))
            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # ax.plot(s[:, 0], s[:, 1], "r")

            # create sample for dataset
            X_traj_i, Y_traj_i = [], []
            for i in range(0, len(c), k):
                x0 = s[i][np.newaxis, :].repeat(k, axis=0)
                # print(x0)

                c_substep = c[i : i + k]
                # print(c_substep)

                t = np.arange(0, config.delta_t_for_step * k, config.delta_t_for_step)[
                    :, np.newaxis
                ]
                # print(t)

                x = np.hstack([x0, c_substep, t])
                # print(x)
                X_traj_i.append(x)

                y = s[i : i + k]
                # print(y)

                Y_traj_i.append(y)
                # exit()

            X.append(np.concatenate(X_traj_i))
            Y.append(np.concatenate(Y_traj_i))

        X = torch.from_numpy(np.stack(X).reshape(-1, config.len_traj, 6))
        Y = torch.from_numpy(np.stack(Y).reshape(-1, config.len_traj, 3))
        return TensorDataset(X, Y)

    @staticmethod
    def generate_data_derivative(robot: Unicycle) -> TensorDataset:
        """
        Generates data
        """
        # Define range of controls
        v_input_range = np.array([-5, +5])
        w_input_range = np.array([-2, +2])

        # Generate random input data
        v_input_data = torch.from_numpy(
            np.random.uniform(v_input_range[0], v_input_range[1], (500_000))
        ).reshape(-1, 1)
        w_input_data = torch.from_numpy(
            np.random.uniform(w_input_range[0], w_input_range[1], (500_000))
        ).reshape(-1, 1)
        random_input_data = torch.hstack([v_input_data, w_input_data])

        # Generate zero input data combinations
        zero_v_input_data = torch.hstack([torch.zeros((5000, 1)), w_input_data[:5000]])
        zero_w_input_data = torch.hstack([v_input_data[:5000], torch.zeros((5000, 1))])
        all_zero_input_data = torch.hstack(
            [torch.zeros((5000, 1)), torch.zeros((5000, 1))]
        )
        zero_input_data = torch.vstack(
            [zero_v_input_data, zero_w_input_data, all_zero_input_data]
        )

        input_data = torch.vstack([random_input_data, zero_input_data])

        # Define range of states
        x_range = np.array([-5, +5])
        y_range = np.array([-5, +5])
        theta_range = np.array([-np.pi, +np.pi])

        # Generate random state data
        x_data = torch.from_numpy(
            np.random.uniform(x_range[0], x_range[1], len(input_data))
        ).reshape(-1, 1)
        y_data = torch.from_numpy(
            np.random.uniform(y_range[0], y_range[1], len(input_data))
        ).reshape(-1, 1)
        theta_data = torch.from_numpy(
            np.random.uniform(theta_range[0], theta_range[1], len(input_data))
        ).reshape(-1, 1)
        state_data = torch.hstack([x_data, y_data, theta_data])

        data = torch.hstack([state_data, input_data]).float()
        labels = robot.torch_f(state_data, input_data).float()

        dataset = TensorDataset(data, labels)
        return dataset

    @staticmethod
    def generate_trajectory_data(robot: Unicycle) -> TensorDataset:

        N_traj = 1000
        traj_len = config.len_traj
        v_input_range = np.array([-3, +3])
        w_input_range = np.array([-np.pi, +np.pi])

        ode = lambda x, u: np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])

        state_buffer = []
        action_buffer = []
        derivative_buffer = []
        next_state_buffer = []

        for _ in range(N_traj):
            V = np.full(
                (traj_len), np.random.uniform(v_input_range[0], v_input_range[1], 1)
            )
            W = np.full(
                (traj_len), np.random.uniform(w_input_range[0], w_input_range[1], 1)
            )
            state0 = np.random.uniform(-5, 5, 3)
            traj_state_buffer = [state0]
            traj_action_buffer = []
            traj_derivative_buffer = []
            traj_next_state_buffer = []

            dt = 0.01
            
            for j in range(traj_len):
                
                traj_action_buffer.append(np.array([V[j], W[j]]))
                traj_derivative_buffer.append(
                    ode(traj_state_buffer[-1], traj_action_buffer[-1])
                )
                traj_next_state_buffer.append(
                    robot.transition(
                        traj_state_buffer[-1], traj_action_buffer[-1], dt
                    )
                    .full()
                    .squeeze()
                )
                traj_state_buffer.append(traj_next_state_buffer[-1])

            traj_state_buffer = np.stack(traj_state_buffer)
            traj_action_buffer = np.stack(traj_action_buffer)
            traj_derivative_buffer = np.stack(traj_derivative_buffer)
            traj_next_state_buffer = np.stack(traj_next_state_buffer)

            state_buffer.append(traj_state_buffer[:-1])
            action_buffer.append(traj_action_buffer)
            derivative_buffer.append(traj_derivative_buffer)
            next_state_buffer.append(traj_next_state_buffer)

        state_buffer = np.stack(state_buffer)
        action_buffer = np.stack(action_buffer)
        next_state_buffer = np.stack(next_state_buffer)

        data = torch.from_numpy(
            np.concatenate([state_buffer, action_buffer], axis=-1)
        ).float()
        labels = torch.from_numpy(next_state_buffer).float()
        return TensorDataset(data, labels)

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

        # 9 zero input + super small noise (+-0,05)
        v = np.zeros((config.N_sample, config.len_traj, 1)) + np.random.uniform(
            -0.05, 0.05, (config.N_sample, config.len_traj, 1)
        )
        w = np.zeros((config.N_sample, config.len_traj, 1)) + np.random.uniform(
            -0.05, 0.05, (config.N_sample, config.len_traj, 1)
        )
        trajectory_input.append(np.concatenate([v, w], axis=-1))

        input_dataset = np.concatenate(trajectory_input, axis=0)

        # make input constant (equal to the first one) for n_step_constant_input steps
        for i in range(0, input_dataset.shape[1], config.n_step_constant_input):
            input_dataset[:, i : i + config.n_step_constant_input, :] = input_dataset[
                :, i, :
            ][:, np.newaxis, :]

        return input_dataset
