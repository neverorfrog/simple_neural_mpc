import casadi as ca
import numpy as np
from matplotlib.axes import Axes

from simple_neural_mpc.models.robot import Robot
from simple_neural_mpc.utils.fancy_vector import FancyVector
from simple_neural_mpc.utils.plotting import plot_wheeled_robot


class Unicycle(Robot):

    @classmethod
    def create_state(cls, *args, **kwargs):
        return UnicycleState(*args, **kwargs)

    @classmethod
    def create_action(cls, *args, **kwargs):
        return UnicycleAction(*args, **kwargs)

    def _init_model(self):

        # state variables
        x, y, psi, t = self.state.variables

        # input variables
        v, w = self.input.variables

        # ODE
        x_dot = v * ca.cos(psi)
        y_dot = v * ca.sin(psi)
        psi_dot = w
        t_dot = 1

        state_dot = ca.vertcat(x_dot, y_dot, psi_dot, t_dot)

        ode = ca.Function("ode", [self.state.syms, self.input.syms], [state_dot])

        integrator = self.integrate(self.state.syms, self.input.syms, ode, self.dt)

        self._transition = ca.Function(
            "transition", [self.state.syms, self.input.syms], [integrator]
        )

    @property
    def transition(self):
        return self._transition

    def plot(self, axis: Axes, state):
        x, y, psi, t = state
        plot_wheeled_robot(axis, x, y, psi)


class UnicycleAction(FancyVector):
    def __init__(self, v=0.0, w=0.0):
        """
        :param a: longitudinal acceleration | [m/s^2]
        :param w: steering angle rate | [rad/s]
        """
        self._values = np.array([v, w])
        self._keys = ["v", "w"]
        self._syms = ca.vertcat(
            *[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))]
        )

    @property
    def v(self):
        return self.values[0]

    @property
    def w(self):
        return self.values[1]

    @v.setter
    def v(self, value: float):
        assert isinstance(value, float)
        self.values[0] = value

    @w.setter
    def w(self, value: float):
        assert isinstance(value, float)
        self.values[1] = value


class UnicycleState(FancyVector):
    def __init__(self, x=0.0, y=0.0, psi=0.0, t=0.0):
        """
        :param x: x coordinate | [m]
        :param y: y coordinate | [m]
        :param psi: yaw angle | [rad]
        """
        self._values = np.array([x, y, psi, t])
        self._keys = ["x", "y", "psi", "t"]
        self._syms = ca.vertcat(
            *[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))]
        )

    @property
    def x(self):
        return self.values[0]

    @property
    def y(self):
        return self.values[1]

    @property
    def psi(self):
        return self.values[2]

    @property
    def t(self):
        return self.values[3]
