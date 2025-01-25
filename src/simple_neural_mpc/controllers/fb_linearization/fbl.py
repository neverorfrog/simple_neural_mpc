import numpy as np
from casadi import cos, sin

from simple_neural_mpc.controllers.controller import Controller
from simple_neural_mpc.models.unicycle_kin.unicycle_kin_casadi import (
    Unicycle,
    UnicycleAction,
)
from simple_neural_mpc.utils.trajectory import Trajectory


class FBL(Controller):
    def __init__(self, kp: np.ndarray, kd: np.ndarray, b=0.1):
        self.kp = kp
        self.kd = kd
        self.b = b

    def command(self, robot: Unicycle, reference: Trajectory):
        state = robot.state
        # point at distance b from center
        x_b = state.x + self.b * cos(state.psi)
        y_b = state.y + self.b * sin(state.psi)

        ref = reference.update(state.t)

        # intermediate control signal
        e_p = ref["p"] - [x_b, y_b]
        u_io = ref["pd"] + self.kp * e_p

        # linearization
        inverse_decoupling_matrix = np.array(
            [
                [cos(state.psi), sin(state.psi)],
                [-sin(state.psi) / self.b, cos(state.psi) / self.b],
            ]
        )

        action = np.matmul(inverse_decoupling_matrix, u_io)
        action = UnicycleAction(action[0], action[1])
        next_state = robot.transition(robot.state.values, action.values).full().squeeze()
        next_state = robot.__class__.create_state(*next_state)
        robot.state = next_state

        return action, next_state, ref["p"], e_p
