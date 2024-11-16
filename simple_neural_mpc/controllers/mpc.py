import casadi as ca
import numpy as np
from omegaconf import OmegaConf

from simple_neural_mpc.controllers.controller import Controller
from simple_neural_mpc.models.differential_drive import (
    DifferentialDrive,
    DifferentialDriveAction,
    DifferentialDriveState,
)

np.random.seed(31)


class ModelPredictiveController(Controller):
    def __init__(self, robot: DifferentialDrive, config: OmegaConf):
        """Optimizer Initialization"""
        self.config = config
        self.robot = robot
        self._init_dims()
        self._init_opti()
        self._init_variables()
        cost = 0
        self.opti.subject_to(
            self.state[:, 0] == self.state0
        )  # constraint on initial state
        for n in range(self.N):
            self._stage_constraints(n)
            cost += self._stage_cost(n)
        cost += self._terminal_cost()
        self.opti.minimize(cost)

    def _init_dims(self):
        # single-track
        self.N = self.config.horizon
        self.dt = self.config.mpc_dt
        self.ns = len(self.robot.state)  # number of state variables
        self.na = len(self.robot.input)  # number of action variables

    def _init_opti(self):
        self.opti = ca.Opti("nlp")
        ipopt_options = {
            "print_level": 2,
            # "linear_solver": "ma27",
            # "hsllib": "/usr/local/lib/libcoinhsl.so",
            # "warm_start_init_point": "yes",
            # "warm_start_bound_push": 1e-8,
            # "nlp_scaling_method": "gradient-based",
            # "nlp_scaling_max_gradient": 100,
        }
        options = {"print_time": False, "expand": True, "ipopt": ipopt_options}
        self.opti.solver("ipopt", options)

    def _init_variables(self):
        # initial state
        self.state0 = self.opti.parameter(self.ns)

        self.state = self.opti.variable(
            self.ns, self.N + 1
        )  # state trajectory var
        self.action = self.opti.variable(
            self.na, self.N
        )  # control trajectory var

        self.state_prediction = np.zeros((self.ns, self.N + 1))
        self.action_prediction = np.ones((self.na, self.N)) + np.random.random(
            (self.na, self.N)
        )

    def _stage_constraints(self, n):
        state: DifferentialDriveState = self.state[:, n]
        action: DifferentialDriveAction = self.action[:, n]

        state_constraints = self.config.state_constraints
        input_constraints = self.config.input_constraints

        # state limits
        # self.opti.subject_to(v >= state_constraints.v_min)
        # self.opti.subject_to(
        #     self.opti.bounded(
        #         state_constraints.delta_min, delta, state_constraints.delta_max
        #     )
        # )

        # input limits
        # self.opti.subject_to(
        #     self.opti.bounded(
        #         input_constraints.a_min, a, input_constraints.a_max
        #     )
        # )
        # self.opti.subject_to(
        #     self.opti.bounded(
        #         input_constraints.w_min, w, input_constraints.w_max
        #     )
        # )

        # Model Dynamics
        self.opti.subject_to(
            self.state[:, n + 1] == self.robot.transition(state, action)
        )

    def _stage_cost(self, n):
        v, delta, s, ey, epsi, t = self._unpack_state(self.state[:, n])
        a, w = self._unpack_action(self.action[:, n])
        ds = self.ds[n]
        cost_weights = self.config.cost_weights
        state_constraints = self.config.state_constraints

        cost = 0

        # cost += ca.if_else(
        #     ey < state_constraints.ey_min,  # violation of road bounds
        #     cost_weights.boundary * ds * (ey - state_constraints.ey_min) ** 2,
        #     0,
        # )

        # cost += ca.if_else(
        #     ey > state_constraints.ey_max,  # violation of road bounds
        #     cost_weights.boundary * ds * (ey - state_constraints.ey_max) ** 2,
        #     0,
        # )

        # cost += (
        #     cost_weights.deviation * ds * (ey**2)
        # )  # deviation from road desciptor

        # cost += cost_weights.w * (w**2)  # steer angle rate

        # if n < self.N - 1:  # Force Action Continuity
        #     next_action = self.action[:, n + 1]
        #     cost += (cost_weights.a) * (
        #         next_action[self.car.input.index("a")] - a
        #     ) ** 2

        # if self.config.obstacles:  # Obstacle avoidance
        #     for obs in self.car.track.obstacles:
        #         distance = ca.sqrt((s - obs.s) ** 2 + (ey - obs.ey) ** 2)
        #         cost += (
        #             cost_weights.obstacles
        #             * ds
        #             / (distance - (obs.radius + 0.1))
        #         )

        return cost

    def _terminal_cost(self):
        cost = 0
        state_constraints = self.config.state_constraints
        cost_weights = self.config.cost_weights
        final_state = self.state[:, -1]
        final_model = self.car
        final_speed = final_state[final_model.state.index("v")]
        cost += ca.if_else(
            final_speed >= state_constraints.v_max,
            cost_weights.v * (final_speed - state_constraints.v_max) ** 2,
            0,
        )  # excessive speed
        cost += cost_weights.time * (
            final_state[final_model.state.index("t"), -1]
        )  # final cost (minimize time)
        cost += (
            cost_weights.ey
            * final_state[final_model.state.index("ey"), -1] ** 2
        )  # final cost (minimize terminal lateral error) hardcodato
        cost += (
            cost_weights.epsi
            * final_state[final_model.state.index("epsi"), -1] ** 2
        )  # final cost (minimize terminal course error) hardcodato
        return cost

    def command(self, state: DifferentialDriveState):
        self._init_horizon(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        action = DifferentialDriveAction(
            v=self.action_prediction[0][0], w=self.action_prediction[1][0]
        )
        return action

    def _init_horizon(self, state: DifferentialDriveState):
        # TODO: complete this method

        # initial state
        state = state.values.squeeze()
        self.opti.set_value(self.state0, state)

        # initializing state and action prediction
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
