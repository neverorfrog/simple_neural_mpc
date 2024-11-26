import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from simple_neural_mpc.controllers.controller import Controller
from simple_neural_mpc.models.unicycle_kin.unicycle_kin_acados_neural import (
    Unicycle,
    UnicycleAction,
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
        self._init_opti()
        self._init_solver()

    def _init_opti(self):
        # OCP
        self.ocp = AcadosOcp()
        self.ocp.model = self.robot.model

        # Dimensions
        self.ns = self.ocp.model.x.rows()
        self.na = self.ocp.model.u.rows()
        self.ocp.dims.nx = self.ns
        self.ocp.dims.nu = self.na
        self.ocp.dims.ny = self.ns + self.na

        # Mapping of variables for cost function
        self.n_opt = self.ns + self.na  # number of optimization variables
        self.n_opt_e = self.ns  # number of optimization variables at the last stage
        Vx = np.zeros((self.n_opt, self.ns))
        Vx[: self.ns, : self.ns] = np.eye(self.ns)
        self.ocp.cost.Vx = Vx  # map state to cost
        Vu = np.zeros((self.n_opt, self.na))
        Vu[self.ns :, : self.na] = np.eye(self.na)
        self.ocp.cost.Vu = Vu
        self.ocp.cost.Vx_e = np.eye(self.ns)  # map state to cost at the last stage

        # Cost
        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"
        self.ocp.cost.W = np.diag(
            np.array([15, 15, 0.01, 1, 1])
        )  # weight matrix for stage cost
        self.ocp.cost.W_e = np.diag(
            np.array([5, 5, 0.01])
        )  # weight matrix for terminal cost

        # Constraints
        x_0, y_0, psi_0 = self.robot.state.values
        self.ocp.constraints.x0 = np.array([x_0, y_0, psi_0])

        # Reference
        self.ocp.cost.yref = np.zeros((self.n_opt))
        self.ocp.cost.yref_e = np.zeros((self.n_opt_e))

        # Solver Options
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_OSQP"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.sim_method_num_stages = 1 # One stage to have explicit euler integration (the model is simple)
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"

        # Prediction Horizon
        self.N = self.config.horizon
        self.ocp.solver_options.tf = self.config.horizon * self.config.dt
        self.ocp.solver_options.N_horizon = self.config.horizon

        # L4Casadi Stuff
        self.ocp.solver_options.model_external_shared_lib_dir = (
            self.robot.l4casadi_model.shared_lib_dir
        )
        self.ocp.solver_options.model_external_shared_lib_name = (
            self.robot.l4casadi_model.name
        )

        # Debug Stuff
        self.ocp.solver_options.print_level = 0

    def _init_solver(self) -> None:
        self.solver = AcadosOcpSolver(self.ocp)
        self.state_prediction = np.zeros((self.ns, self.N + 1))
        self.action_prediction = np.zeros((self.na, self.N))
        for n in range(self.N + 1):
            self.solver.set(n, "x", self.state_prediction[:, n])
        for n in range(self.N):
            self.solver.set(n, "u", self.action_prediction[:, n])

    def command(self, robot: Unicycle, reference: Trajectory):

        # generate trajectory for next N steps
        t = np.linspace(
            self.k * self.config.dt,
            (self.k + self.N) * self.config.dt,
            self.N + 1,
        )
        ref = reference.update(t)
        pos = ref["p"]
        psi = ref["psi"]
        pd = ref["pd"]
        w = ref["psid"]
        v = np.sqrt(np.sum(pd**2, axis=0))
        for j in range(self.N):
            self.solver.set(
                j, "yref", np.array([pos[0, j], pos[1, j], psi[j], v[j], w[j]])
            )
        self.solver.set(
            self.N, "yref", np.array([pos[0, self.N], pos[1, self.N], psi[self.N]])
        )
        self.k += 1

        # Solve the optimization problem
        action = self.solver.solve_for_x0(robot.state.values)
        action = UnicycleAction(v=action[0], w=action[1])
        robot.input = action

        print(action)

        next_state = self.solver.get(1, "x")
        error = np.linalg.norm(next_state[:2] - pos[:, 0])
        next_state = robot.__class__.create_state(*next_state)
        robot.state = next_state

        print(next_state)
        print("Error: ", error)
        print("")

        return action, next_state, pos, error
