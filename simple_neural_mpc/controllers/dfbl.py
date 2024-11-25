import casadi as ca
import numpy as np
from casadi import cos, sin

from simple_neural_mpc.controllers.controller import Controller
from simple_neural_mpc.models.differential_drive_kin import (
    DifferentialDrive,
    DifferentialDriveAction,
)
from simple_neural_mpc.utils.trajectory import Trajectory


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

    def command(self, robot: DifferentialDrive, reference: Trajectory):
        state = robot.state

        # calculating velocity
        input = robot.input
        xd = cos(state.psi) * input.v
        yd = sin(state.psi) * input.v

        ref = reference.update(state.t)
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
        return DifferentialDriveAction(v, a_w[1]), ref["p"], e_p

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
