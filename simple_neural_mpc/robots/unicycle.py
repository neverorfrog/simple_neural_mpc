import casadi as ca
import l4casadi as l4c
import numpy as np
import torch
from acados_template import AcadosModel
from matplotlib.axes import Axes

from simple_neural_mpc.robots.robot import Robot
from simple_neural_mpc.simulation.plotting import plot_wheeled_robot
from simple_neural_mpc.utils.fancy_vector import FancyVector
from simple_neural_mpc.config.mpc_config import MPCConfig as config
from simple_neural_mpc.utils.integrators import RK2


class Unicycle(Robot):
    
    def __init__(self, neural_network : torch.nn.Module = None):
        self.neural_network = neural_network
        super().__init__()

    def _init_model(self):
        
        dt = ca.MX.sym("dt")
        model_name = config.model_name

        # state variables
        x, y, psi = self.state.variables
        state = ca.vertcat(x, y, psi)

        # input variables
        v, w = self.input.variables
        control = ca.vertcat(v, w)

        # state dot variables
        x_dot, y_dot, psi_dot = (
            ca.MX.sym("x_dot"),
            ca.MX.sym("y_dot"),
            ca.MX.sym("psi_dot"),
        )
        state_dot = ca.vertcat(x_dot, y_dot, psi_dot)

        # Explicit ODE
        x_dot = v * ca.cos(psi)
        y_dot = v * ca.sin(psi)
        psi_dot = w
        f_expl = ca.vertcat(x_dot, y_dot, psi_dot)
        self.integrator = RK2(state, control, f_expl, dt)

        # Implicit ODE
        f_impl = state_dot - f_expl

        # Create acados model
        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = state
        model.xdot = state_dot
        model.u = control
        model.z = ca.vertcat([])
        model.p = ca.vertcat([])
        model.t_label = "$t$ [s]"
        model.x_labels = ["$x$ [m]", "$y$ [m]", "$\psi$ [rad]"]
        model.u_labels = ["$v$ [m/s]", "$\omega$ [rad/s]"]
        
        
        if config.neural is True and self.neural_network is not None:
            self.l4casadi_model = l4c.L4CasADi(self.neural_network, name=model_name)
            neural_dyn = self.l4casadi_model(
                ca.vertcat(state, control, config.dt)
            )  # neural network approximated dynamics (MX)
            f_disc = neural_dyn
            model.disc_dyn_expr = f_disc
            

    @property
    def transition(self):
        return self.integrator.step
        
    def plot(self, axis: Axes, state):
        x, y, psi = state
        plot_wheeled_robot(axis, x, y, psi)
        
    @classmethod
    def create_state(cls, *args, **kwargs):
        return UnicycleState(*args, **kwargs)

    @classmethod
    def create_action(cls, *args, **kwargs):
        return UnicycleAction(*args, **kwargs)


class UnicycleAction(FancyVector):
    def __init__(self, v=0.0, w=0.0):
        """
        :param a: longitudinal acceleration | [m/s^2]
        :param w: steering angle rate | [rad/s]
        """
        self._values = np.array([v, w])
        self._keys = ["v", "w"]
        self._syms = ca.vertcat(
            *[ca.MX.sym(self._keys[i]) for i in range(len(self._keys))]
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
        self._values = np.array([x, y, psi])
        self._keys = ["x", "y", "psi"]
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
