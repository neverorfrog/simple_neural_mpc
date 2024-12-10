import numpy as np

'''

In this file we generate trajectory using a stupid model

''' 

# Explicit ODE
class Unicycle():
    
    def __init__(self, x0, y0, theta0):
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.dt = 0.1
        self.state_buffer = np.empty([0, 3], dtype=np.float32)
        self.action_buffer = np.empty([0, 2], dtype=np.float32)

        self.state_buffer = np.vstack([self.state_buffer, [x0, y0, theta0]])

    def unicycle_forward(self, w, v):
        x_dot = v * np.cos(self.theta)
        y_dot = v * np.sin(self.theta)
        psi_dot = w
        
        self.x = self.x + x_dot * self.dt
        self.y = self.y + y_dot * self.dt
        self.theta = self.theta + psi_dot * self.dt
        
        self.update_buffer(v, w)
        
    def update_buffer(self, v, w):
        action = [v, w]
        new_state = [self.x, self.y, self.theta]
        
        self.state_buffer = np.vstack([self.state_buffer, new_state])
        self.action_buffer = np.vstack([self.action_buffer, action])
        
    def get_state(self):
    	return self.x, self.y, self.theta




if __name__ == '__main__': 


    import os
    import matplotlib.pyplot as plt


    print(os.getcwd())

    
    n_step = 1000  # aka number of action
    n_traj = 10 
    plot_traj = False
    destination_path = 'neural_model_identification/data_module/trajectories/'
    for traj_id in range(n_traj):

        seed = np.linspace(0, 100, 1000)
        V = np.random.randn(n_step) + 4
        W = np.sin(seed) + np.random.randn(n_step)*0.7
        
        x0, y0, theta0 = np.random.rand(3)
        unicycle = Unicycle(x0, y0, theta0)
        for i in range(n_step):
            unicycle.unicycle_forward(W[i], V[i])

        state_buffer = unicycle.state_buffer
        action_buffer = unicycle.action_buffer
        os.makedirs(destination_path + f"sample_{traj_id}", exist_ok=True)
        np.save(destination_path + f'sample_{traj_id}/actions.npy', action_buffer)
        np.save(destination_path + f'sample_{traj_id}/states.npy', state_buffer)

        print(f'Trajectory {traj_id} generated and saved')
        
        if plot_traj:
            plt.plot(state_buffer[:, 0], state_buffer[:, 1])    
            plt.show()