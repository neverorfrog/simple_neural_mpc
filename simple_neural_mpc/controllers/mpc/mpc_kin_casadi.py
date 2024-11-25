import casadi as ca
import numpy as np

from simple_neural_mpc.controllers.controller import Controller
from simple_neural_mpc.models.unicycle_kin.unicycle_kin_casadi import (
    Unicycle,
    UnicycleAction,
    UnicycleState,
)
from simple_neural_mpc.utils.configuration import (
    KinModelPredictiveControllerConfig,
)
from simple_neural_mpc.utils.trajectory import Trajectory

np.random.seed(31)


class ModelPredictiveController(Controller):
    def __init__(self, robot: Unicycle, config: KinModelPredictiveControllerConfig):
        """Optimizer Initialization"""
        self.config = config
        self.robot = robot
        self.k = 0  # current iteration
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
        self.N = self.config.horizon
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

        self.state = self.opti.variable(self.ns, self.N + 1)  # state trajectory var
        self.action = self.opti.variable(self.na, self.N)  # control trajectory var

        self.state_prediction = np.zeros((self.ns, self.N + 1))
        self.action_prediction = np.ones((self.na, self.N)) + np.random.random(
            (self.na, self.N)
        )

        self.ref = self.opti.parameter(2, self.N + 1)

    def _stage_constraints(self, n):
        state = self.state[:, n]
        action = self.action[:, n]
        v, w = self._unpack_action(action)

        input_constraints = self.config.input_constraints

        # input limits
        self.opti.subject_to(
            self.opti.bounded(input_constraints.v_min, v, input_constraints.v_max)
        )
        self.opti.subject_to(
            self.opti.bounded(input_constraints.w_min, w, input_constraints.w_max)
        )

        # Model Dynamics
        self.opti.subject_to(self.state[:, n + 1] == self.robot.transition(state, action))

    def _stage_cost(self, n):
        v, w = self._unpack_action(self.action[:, n])
        cost_weights = self.config.cost_weights
        cost = 0

        # Mean Squared Error on trajectory
        cost += self.config.cost_weights.ex * ca.sumsqr(
            self.state[:2, :-1] - self.ref[:, :-1]
        )  # MSE on position

        # Control minimization
        cost += cost_weights.w * (w**2)  # steer angle rate
        cost += cost_weights.v * (v**2)

        return cost

    def _terminal_cost(self):
        # Mean Squared Error on trajectory final point
        cost = 10 * ca.sumsqr(self.state[:2, -1] - self.ref[:, -1])  # MSE on position

        return cost

    def command(self, robot: Unicycle, reference: Trajectory):
        self._init_horizon(robot.state)

        # generate trajectory for next N steps
        t = np.linspace(
            self.k * self.config.dt,
            (self.k + self.N) * self.config.dt,
            self.N + 1,
        )
        ref = reference.update(t)["p"]
        self.opti.set_value(self.ref, ref)
        self.k += 1

        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        action = UnicycleAction(
            v=self.action_prediction[0][0], w=self.action_prediction[1][0]
        )
        robot.input = action

        next_state = robot.transition(robot.state.values, action.values).full().squeeze()
        next_state = robot.__class__.create_state(*next_state)
        error = np.linalg.norm(self.state_prediction[:2, 0] - ref[:2, 0])
        robot.state = next_state

        return action, next_state, ref, error

    def _init_horizon(self, state: UnicycleState):
        # initial state
        state = state.values.squeeze()
        self.opti.set_value(self.state0, state)

        # initializing state and action prediction
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)

    def _unpack_state(self, state: np.ndarray):
        x = state[self.robot.state.index("x")]
        y = state[self.robot.state.index("y")]
        psi = state[self.robot.state.index("psi")]
        return x, y, psi

    def _unpack_action(self, action: np.ndarray):
        v = action[self.robot.input.index("v")]
        w = action[self.robot.input.index("w")]
        return v, w
