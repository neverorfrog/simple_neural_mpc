import numpy as np
import os
import matplotlib.pyplot as plt
from neural_model_identification.models.unicycle_dyn import Unicycle

if __name__ == '__main__': 
    # print(os.getcwd())

    n_step = 1000  # aka number of action
    n_traj = 10 
    plot_traj = True
    destination_path = '../data_module/trajectories_dyn/'
    for traj_id in range(n_traj):

        seed = np.linspace(0, 100, 1000)
        F_l = np.random.randn(n_step) + 4
        F_r = np.random.randn(n_step)*0.86 + 2
        
        x0, y0, theta0, v0, w0 = np.random.rand(5)
        unicycle = Unicycle(x0, y0, theta0, v0, w0)
        for i in range(n_step):
            unicycle.unicycle_forward(F_l[i], F_r[i])

        state_buffer = unicycle.state_buffer
        action_buffer = unicycle.action_buffer
        os.makedirs(destination_path + f"sample_{traj_id}", exist_ok=True)
        np.save(destination_path + f'sample_{traj_id}/actions.npy', action_buffer)
        np.save(destination_path + f'sample_{traj_id}/states.npy', state_buffer)

        print(f'Trajectory {traj_id} generated and saved')
        
        if plot_traj:
            plt.plot(state_buffer[:, 0], state_buffer[:, 1])    
            plt.show()