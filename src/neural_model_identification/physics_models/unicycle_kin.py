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
        self.t = 0
        self.state_buffer = np.empty([0, 3], dtype=np.float32)
        self.action_buffer = np.empty([0, 2], dtype=np.float32)
        self.derivative_buffer = np.empty([0, 3], dtype=np.float32)
        self.time_buffer = np.empty([0, 1], dtype=np.float32)

        self.state_buffer = np.vstack([self.state_buffer, [x0, y0, theta0]])
        self.time_buffer = np.vstack([self.time_buffer, [self.t]])

    def unicycle_forward(self, w, v):
        self.x_dot = v * np.cos(self.theta)
        self.y_dot = v * np.sin(self.theta)
        self.theta_dot = w

        self.x = self.x + self.x_dot * self.dt
        self.y = self.y + self.y_dot * self.dt
        self.theta = self.theta + self.theta_dot * self.dt

        self.t += self.dt

        self.update_buffer(v, w)

    def update_buffer(self, v, w):
        self.state_buffer = np.vstack([self.state_buffer, [self.x, self.y, self.theta]])
        self.action_buffer = np.vstack([self.action_buffer, [v, w]])
        self.derivative_buffer = np.vstack(
            [self.derivative_buffer, [self.x_dot, self.y_dot, self.theta_dot]]
        )
        self.time_buffer = np.vstack([self.time_buffer, [self.t]])

    def get_state(self):
        return self.x, self.y, self.theta
