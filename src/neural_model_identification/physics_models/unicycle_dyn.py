import numpy as np

from simple_neural_mpc.utils.configuration import UnicycleConfig
from simple_neural_mpc.utils.misc import load_config, project_root

"""

In this file we implement kinematic unicycle to generate trajectories for training phase

"""


# Explicit ODE
class Unicycle:

    def __init__(self, x0, y0, theta0, v0, w0):
        self.config = load_config(
            f"{project_root()}/config/models/unicycle.yaml",
            UnicycleConfig,
        )

        self.x = x0
        self.y = y0
        self.theta = theta0
        self.v = v0
        self.w = w0
        self.dt = 0.1
        self.m = self.config.car.m
        self.l = self.config.car.l

        self.state_buffer = np.empty([0, 5], dtype=np.float32)
        self.action_buffer = np.empty([0, 2], dtype=np.float32)

        self.state_buffer = np.vstack([self.state_buffer, [x0, y0, theta0, v0, w0]])

    def unicycle_forward(self, F_l, F_r):
        x_dot = self.v * np.cos(self.theta)
        y_dot = self.v * np.sin(self.theta)
        psi_dot = self.w
        v_dot = (F_l + F_r) / self.m
        w_dot = (F_r - F_l) / (self.m * self.l)

        self.x = self.x + x_dot * self.dt
        self.y = self.y + y_dot * self.dt
        self.theta = self.theta + psi_dot * self.dt
        self.v = self.v + v_dot * self.dt
        self.w = self.w + w_dot * self.dt

        self.update_buffer(F_l, F_r)

    def update_buffer(self, F_l, F_r):
        action = [F_l, F_r]
        new_state = [self.x, self.y, self.theta, self.v, self.w]

        self.state_buffer = np.vstack([self.state_buffer, new_state])
        self.action_buffer = np.vstack([self.action_buffer, action])

    def get_state(self):
        return self.x, self.y, self.theta, self.v, self.w
