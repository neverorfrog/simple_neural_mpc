from typing import Tuple
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from simple_neural_mpc.config.mpc_config import MPCConfig
from simple_neural_mpc.config.neural_config import DatasetConfig as config
from simple_neural_mpc.neural_modeling.dataset.tensor_dataset import (
    TensorDataset,
)
from simple_neural_mpc.robots.unicycle import Unicycle


class UnicycleDataset:

    @staticmethod
    def generate_trajectory_data(robot: Unicycle) -> TensorDataset:

        N_traj = config.N_traj
        traj_len = config.len_traj

        state_buffer = []
        action_buffer = []
        next_state_buffer = []
        time_buffer = []


        for i in range(N_traj):

            dt = config.delta_t_for_step

            state0 = np.random.uniform(-2, 2, 3)
            traj_state_buffer = []
            start_state_buffer = [state0]
            traj_action_buffer = []
            traj_next_state_buffer = []
            traj_time_buffer = []

            sub_traj_len = config.n_step_constant_input

            for _ in range(0, traj_len, sub_traj_len):
                V, W = generate_input_data(i, sub_traj_len)
                for k in range(sub_traj_len):
                    traj_action_buffer.append(np.array([V[k], W[k]]))
                    next_state = (
                        robot.transition(
                            start_state_buffer[-1], traj_action_buffer[-1], dt
                        )
                        .full()
                        .squeeze()
                    )
                    traj_next_state_buffer.append(next_state)
                    traj_state_buffer.append(state0)
                    start_state_buffer.append(next_state)
                    traj_time_buffer.append((k + 1) * dt)

                state0 = start_state_buffer[-1]

            traj_state_buffer = np.stack(traj_state_buffer)
            traj_action_buffer = np.stack(traj_action_buffer)
            traj_next_state_buffer = np.stack(traj_next_state_buffer)
            traj_time_buffer = np.stack(traj_time_buffer).reshape(-1, 1)

            state_buffer.append(traj_state_buffer)
            action_buffer.append(traj_action_buffer)
            next_state_buffer.append(traj_next_state_buffer)
            time_buffer.append(traj_time_buffer)

        state_buffer = np.stack(state_buffer)
        action_buffer = np.stack(action_buffer)
        next_state_buffer = np.stack(next_state_buffer)
        time_buffer = np.stack(time_buffer)

        data = torch.from_numpy(
            np.concatenate([state_buffer, action_buffer, time_buffer], axis=-1)
        ).float()
        labels = torch.from_numpy(next_state_buffer).float()
        return TensorDataset(data, labels)
    
    
    @staticmethod
    def generate_derivative_data(robot: Unicycle) -> TensorDataset:

        N_traj = config.N_traj
        traj_len = config.len_traj

        state_buffer = []
        action_buffer = []
        derivative_buffer = []

        for i in range(N_traj):

            dt = config.delta_t_for_step

            state0 = np.random.uniform(-2, 2, 3)
            traj_state_buffer = [state0]
            traj_action_buffer = []
            traj_derivative_buffer = []

            V, W = generate_input_data(i, traj_len)
            for k in range(traj_len):
                traj_action_buffer.append(np.array([V[k], W[k]]))
                derivative = robot.f_expl(0, traj_state_buffer[-1], traj_action_buffer[-1])
                next_state = (
                    robot.transition(
                        traj_state_buffer[-1], traj_action_buffer[-1], dt
                    )
                    .full()
                    .squeeze()
                )
                traj_state_buffer.append(next_state)
                traj_derivative_buffer.append(derivative)


            traj_state_buffer = np.stack(traj_state_buffer[:-1])
            traj_action_buffer = np.stack(traj_action_buffer)
            traj_derivative_buffer = np.stack(traj_derivative_buffer)

            state_buffer.append(traj_state_buffer)
            action_buffer.append(traj_action_buffer)
            derivative_buffer.append(traj_derivative_buffer)

        state_buffer = np.stack(state_buffer)
        action_buffer = np.stack(action_buffer)
        derivative_buffer = np.stack(derivative_buffer)

        data = torch.from_numpy(
            np.concatenate([state_buffer, action_buffer], axis=-1)
        ).float()
        labels = torch.from_numpy(derivative_buffer).float()
        return TensorDataset(data, labels)

    @staticmethod
    def generate_test_data(robot: Unicycle) -> TensorDataset:

        N_traj = 100
        traj_len = 50

        state_buffer = []
        action_buffer = []
        next_state_buffer = []
        time_buffer = []

        for i in range(N_traj):

            state0 = np.random.uniform(-2, 2, 3)
                
            V, W = generate_input_data(i, traj_len)

            traj_state_buffer = [state0]
            traj_action_buffer = []
            traj_next_state_buffer = []
            traj_time_buffer = []

            dt = 0.01

            for j in range(traj_len):
                traj_action_buffer.append(np.array([V[j], W[j]]))
                next_state = (
                    robot.transition(traj_state_buffer[-1], traj_action_buffer[-1], dt)
                    .full()
                    .squeeze()
                )
                traj_next_state_buffer.append(next_state)
                traj_state_buffer.append(next_state)
                traj_time_buffer.append(dt)

            traj_state_buffer = np.stack(traj_state_buffer)
            traj_action_buffer = np.stack(traj_action_buffer)
            traj_next_state_buffer = np.stack(traj_next_state_buffer)
            traj_time_buffer = np.stack(traj_time_buffer).reshape(-1, 1)

            state_buffer.append(traj_state_buffer[:-1])
            action_buffer.append(traj_action_buffer)
            next_state_buffer.append(traj_next_state_buffer)
            time_buffer.append(traj_time_buffer)

        state_buffer = np.stack(state_buffer)
        action_buffer = np.stack(action_buffer)
        next_state_buffer = np.stack(next_state_buffer)
        traj_time_buffer = np.stack(time_buffer)

        data = torch.from_numpy(
            np.concatenate([state_buffer, action_buffer, time_buffer], axis=-1)
        ).float()
        labels = torch.from_numpy(next_state_buffer).float()
        return TensorDataset(data, labels)
    
    
def generate_input_data(i: int, sub_traj_len: int) -> Tuple[np.ndarray]:
    v_input_range = np.array([-5, +5])
    w_input_range = np.array([-3, +3])
    eps = 1e-4
    if i % 30 < eps:  # every 30 a trajectory of all zeros
        v = 0
        w = 0
    elif i % 10 < eps:  # every 10 a pure rotation
        v = 0
        w = np.random.uniform(w_input_range[0], w_input_range[1], 1)
    elif i % 5 < eps:  # every 5 a straight trajectory
        v = np.random.uniform(v_input_range[0], v_input_range[1], 1)
        w = 0
    else:  # the rest are random
        v = np.random.uniform(v_input_range[0], v_input_range[1], 1)
        w = np.random.uniform(w_input_range[0], w_input_range[1], 1)

    V = np.full((sub_traj_len), v)
    W = np.full((sub_traj_len), w)
    return V, W