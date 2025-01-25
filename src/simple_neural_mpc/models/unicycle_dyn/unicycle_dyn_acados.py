import casadi as ca
import numpy as np
from acados_template import AcadosModel
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
        model_name = "dynamic_unicycle"

        # state variables
        x, y, psi, v, w = self.state.variables
        state = ca.vertcat(x, y, psi, v, w)

        # input variables
        F_l, F_r = self.input.variables
        control = ca.vertcat(F_l, F_r)

        # config variables
        m = self.config.car.m
        l = self.config.car.l

        # state dot variables
        x_dot, y_dot, psi_dot, v_dot, w_dot = (
            ca.MX.sym("x_dot"),
            ca.MX.sym("y_dot"),
            ca.MX.sym("psi_dot"),
            ca.MX.sym("v_dot"),
            ca.MX.sym("w_dot"),
        )
        state_dot = ca.vertcat(x_dot, y_dot, psi_dot, v_dot, w_dot)

        # Explicit ODE
        x_dot = v * ca.cos(psi)
        y_dot = v * ca.sin(psi)
        psi_dot = w
        v_dot = (F_l + F_r) / (2 * m)
        w_dot = (F_r - F_l) / (m * l)
        f_expl = ca.vertcat(x_dot, y_dot, psi_dot, v_dot, w_dot)

        # Implicit ODE
        f_impl = state_dot - f_expl

        # Create acados model
        self.model = AcadosModel()
        self.model.name = model_name
        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = state
        self.model.xdot = state_dot
        self.model.u = control
        self.model.t_label = "$t$ [s]"
        self.model.x_labels = [
            "$x$ [m]",
            "$y$ [m]",
            "$\psi$ [rad]",
            "$v$ [m/s]",
            "$\omega$ [rad/s]",
        ]
        self.model.u_labels = ["$F_l$ [N]", "$F_r$ [N]"]

    @property
    def transition(self):
        pass

    def plot(self, axis: Axes, state):
        x, y, psi, _, _ = state
        plot_wheeled_robot(axis, x, y, psi)


class UnicycleState(FancyVector):
    def __init__(self, x=0.0, y=0.0, psi=0.0, v=0.0, w=0.0, t=0.0):
        """
        :param x: x coordinate | [m]
        :param y: y coordinate | [m]
        :param psi: yaw angle | [rad]
        :param v: velocity | [m/s]
        :param w: angular velocity | [rad/s]
        """
        self._values = np.array([x, y, psi, v, w])
        self._keys = ["x", "y", "psi", "v", "w"]
        self._syms = ca.vertcat(
            *[ca.MX.sym(self._keys[i]) for i in range(len(self._keys))]
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
    def v(self):
        return self.values[3]

    @property
    def w(self):
        return self.values[4]


class UnicycleAction(FancyVector):
    def __init__(self, F_l=0.0, F_r=0.0):
        """
        :param F_l: force on left wheel  | [N]
        :param F_r: force on right wheel | [N]
        """
        self._values = np.array([F_l, F_r])
        self._keys = ["F_l", "F_r"]
        self._syms = ca.vertcat(
            *[ca.MX.sym(self._keys[i]) for i in range(len(self._keys))]
        )

    @property
    def F_l(self):
        return self.values[0]

    @property
    def F_r(self):
        return self.values[1]

    @F_l.setter
    def F_l(self, value: float):
        assert isinstance(value, float)
        self.values[0] = value

    @F_r.setter
    def F_r(self, value: float):
        assert isinstance(value, float)
        self.values[1] = value
