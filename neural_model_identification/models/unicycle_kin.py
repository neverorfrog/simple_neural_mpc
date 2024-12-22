import numpy as np

'''

In this file we implement kinematic unicycle to generate trajectories for training phase

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

