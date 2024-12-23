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
        seed = np.linspace(0, 100, 1000)
        V = np.random.randn(n_step) + 2
        W = np.sin(seed) + np.random.randn(n_step) * 1.0

        x0, y0, theta0 = np.random.rand(3)
        unicycle = Unicycle(x0, y0, theta0)
        for i in range(n_step):
            unicycle.unicycle_forward(W[i], V[i])

        state_buffer = unicycle.state_buffer
        action_buffer = unicycle.action_buffer
        derivative_buffer = unicycle.derivative_buffer
        os.makedirs(os.path.join(destination_path, f"sample_{traj_id}"), exist_ok=True)
        np.save(
            os.path.join(destination_path, f"sample_{traj_id}/actions.npy"), action_buffer
        )
        np.save(
            os.path.join(destination_path, f"sample_{traj_id}/states.npy"), state_buffer
        )
        np.save(
            os.path.join(destination_path, f"sample_{traj_id}/derivatives.npy"),
            derivative_buffer,
        )

        print(f"Trajectory {traj_id} generated and saved")

        if plot_traj:
            plt.plot(state_buffer[:, 0], state_buffer[:, 1])
            plt.show()
