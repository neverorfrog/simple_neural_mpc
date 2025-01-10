import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from neural_model_identification.parameters.train_params import TrainParams
from neural_model_identification.physics_models.unicycle_kin import Unicycle

if __name__ == "__main__":
    n_traj = TrainParams.n_traj
    n_step = TrainParams.n_step
    plot_traj = TrainParams.plot_traj
    destination_path = TrainParams.data_path

    try:
        shutil.rmtree(destination_path)
    except FileNotFoundError:
        pass

    for traj_id in range(n_traj):
        
        V = np.full(n_step, np.random.rand() * 5.0)
        W = np.full(n_step, np.random.rand() * 2 * np.pi) 
        
        x0, y0, theta0 = np.random.rand(3)
        unicycle = Unicycle(x0, y0, theta0)
        for i in range(n_step):
            unicycle.unicycle_forward(W[i], V[i])

        os.makedirs(os.path.join(destination_path, f"sample_{traj_id}"), exist_ok=True)
        np.save(
            os.path.join(destination_path, f"sample_{traj_id}/actions.npy"),
            unicycle.action_buffer,
        )
        np.save(
            os.path.join(destination_path, f"sample_{traj_id}/states.npy"),
            unicycle.state_buffer,
        )
        np.save(
            os.path.join(destination_path, f"sample_{traj_id}/derivatives.npy"),
            unicycle.derivative_buffer,
        )
        np.save(
            os.path.join(destination_path, f"sample_{traj_id}/times.npy"),
            unicycle.time_buffer,
        )

        print(f"Trajectory {traj_id} generated and saved")
