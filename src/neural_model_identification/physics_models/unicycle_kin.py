import numpy as np

from neural_model_identification.parameters.train_params import TrainParams

"""

In this file we implement kinematic unicycle to generate trajectories for training phase

"""


# Explicit ODE
class Unicycle:

    def __init__(self, x0, y0, theta0):
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.dt = TrainParams.dt
        self.state_buffer = np.empty([0, 3], dtype=np.float32)
        self.action_buffer = np.empty([0, 2], dtype=np.float32)
        self.derivative_buffer = np.empty([0, 3], dtype=np.float32)

        self.state_buffer = np.vstack([self.state_buffer, [x0, y0, theta0]])

    def unicycle_forward(self, w, v):
        self.x_dot = v * np.cos(self.theta)
        self.y_dot = v * np.sin(self.theta)
        self.theta_dot = w

        self.x = self.x + self.x_dot * self.dt
        self.y = self.y + self.y_dot * self.dt
        self.theta = self.theta + self.theta_dot * self.dt

        self.update_buffer(v, w)

    def update_buffer(self, v, w):
        action = [v, w]
        new_state = [self.x, self.y, self.theta]
        new_derivative = [self.x_dot, self.y_dot, self.theta_dot]
        self.state_buffer = np.vstack([self.state_buffer, new_state])
        self.action_buffer = np.vstack([self.action_buffer, action])
        self.derivative_buffer = np.vstack([self.derivative_buffer, new_derivative])

    def get_state(self):
        return self.x, self.y, self.theta
