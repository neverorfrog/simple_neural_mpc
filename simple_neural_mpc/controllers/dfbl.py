import casadi as ca
import numpy as np
from casadi import cos, sin

from simple_neural_mpc.controllers.controller import Controller
from simple_neural_mpc.robots.unicycle import Unicycle, UnicycleAction
from simple_neural_mpc.simulation.trajectory import Trajectory
from simple_neural_mpc.config.mpc_config import MPCConfig as config


class DFBL(Controller):
    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.kd = kd
        v = ca.SX.sym("v")
        a = ca.SX.sym("a")
        v_dot = a
        self.ode = ca.Function("ode", [v, a], [v_dot])
        integrator = self.integrate(v, a, h=0.05)
        self.v_transition = ca.Function("transition", [v, a], [integrator])
        self.t = 0

    def command(self, robot: Unicycle, reference: Trajectory, t: float):
        state = robot.state

        # calculating velocity
        input = robot.input
        xd = cos(state.psi) * input.v
        yd = sin(state.psi) * input.v

        ref = reference.update(t)
        e_p = ref["p"] - [state.x, state.y]
        e_d = ref["pd"] - [xd, yd]
        u_io = ref["pdd"] + e_p * self.kp + e_d * self.kd

        inverse_decoupling_matrix = np.array(
            [
                [cos(state.psi), sin(state.psi)],
                [-sin(state.psi) / input.v, cos(state.psi) / input.v],
            ]
        )

        a_w = np.matmul(inverse_decoupling_matrix, u_io)
        v = self.v_transition(input.v, a_w[0]).full().squeeze()

        action = UnicycleAction(v, a_w[1])
        next_state = robot.transition(robot.state.values, action.values, config.dt).full().squeeze()
        next_state = robot.__class__.create_state(*next_state)
        robot.state = next_state

        print(ref["p"])

        return action, next_state, ref["p"], e_p

    def integrate(self, v, a, h):
        """
        RK4 integrator
        h: integration interval
        """
        vd_1 = self.ode(v, a)
        vd_2 = self.ode(v + (h / 2) * vd_1, a)
        vd_3 = self.ode(v + (h / 2) * vd_2, a)
        vd_4 = self.ode(v + h * vd_3, a)
        new_v = v + (1 / 6) * (vd_1 + 2 * vd_2 + 2 * vd_3 + vd_4) * h
        return new_v
