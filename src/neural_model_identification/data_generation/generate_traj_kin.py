import sys
sys.path.append('.')

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np


#from neural_model_identification.parameters.train_params import TrainParams
from neural_model_identification.parameters.train_params_pinn import TrainParamsPinn as TrainParams
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
        
        if traj_id % 13 == 0:
            lin_val = 0
            ang_val = 0
        if traj_id % 19 == 0:
            lin_val = np.random.rand()
            ang_val = 0
        if traj_id % 16 == 0:
            lin_val = 0
            ang_val = np.random.rand()
        else:
            lin_val = np.random.rand()
            ang_val = np.random.rand()
        
        V = np.full(n_step, lin_val)
        W = np.full(n_step, ang_val) 
        
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
