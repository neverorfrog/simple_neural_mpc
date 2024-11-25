import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
from simple_neural_mpc.controllers.controller import Controller
from simple_neural_mpc.models.differential_drive_kin_acados import (
    DifferentialDrive,
    DifferentialDriveAction,
    DifferentialDriveState,
)
from simple_neural_mpc.utils.configuration import (
    KinModelPredictiveControllerConfig,
)
from simple_neural_mpc.utils.trajectory import Trajectory

np.random.seed(31)

class ModelPredictiveController(Controller):
    def __init__(self, robot: DifferentialDrive, config: KinModelPredictiveControllerConfig):
        """Optimizer Initialization"""
        self.config = config
        self.robot = robot
        self.k = 0  # current iteration
        self._init_opti()
        self._init_solver()
        self._init_integrator()
        
    def _init_opti(self):
        self.ocp = AcadosOcp() 
        self.ocp.model = self.robot.model
        
        # Dimensions
        self.ns = self.ocp.model.x.rows()
        self.na = self.ocp.model.u.rows()
        
        # Mapping of variables for cost function
        self.n_opt = self.ns + self.na # number of optimization variables
        self.n_opt_e = self.ns # number of optimization variables at the last stage
        Vx = np.zeros((self.n_opt, self.ns))
        Vx[:self.ns, :self.ns] = np.eye(self.ns)
        self.ocp.cost.Vx = Vx # map state to cost
        Vu = np.zeros((self.n_opt, self.na)) 
        Vu[self.ns:, :self.na] = np.eye(self.na)
        self.ocp.cost.Vu = Vu
        self.ocp.cost.Vx_e = np.eye(self.ns) # map state to cost at the last stage
        
        # Cost
        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"
        self.ocp.cost.W = np.diag(np.array([15, 15, 0.01, 1, 1])) # weight matrix for stage cost
        self.ocp.cost.W_e = np.diag(np.array([5, 5, 0.01])) # weight matrix for terminal cost
        
        # Constraints
        x_0, y_0, psi_0 = self.robot.state.values
        self.ocp.constraints.x0 = np.array([x_0, y_0, psi_0])
        # self.ocp.constraints.lbu = np.array([-self.config.input_constraints.v_max, -self.config.input_constraints.w_max])
        # self.ocp.constraints.ubu = np.array([self.config.input_constraints.v_max, self.config.input_constraints.w_max])
        # self.ocp.constraints.idxbu = np.array([0, 0])
        
        # Reference
        self.ocp.cost.yref = np.zeros((self.n_opt))
        self.ocp.cost.yref_e = np.zeros((self.n_opt_e))
        
        # Solver Options
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.integrator_type = "IRK"
        self.ocp.solver_options.nlp_solver_type = "SQP"
        
        # Prediction Horizon
        self.N = self.config.horizon
        self.ocp.solver_options.tf = self.config.horizon * self.config.dt
        self.ocp.solver_options.N_horizon = self.config.horizon
        
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
            
    def _init_integrator(self) -> None:
        self.integrator = AcadosSimSolver(self.ocp)
        
    def command(self, robot: DifferentialDrive, reference: Trajectory):
        # self._init_horizon(robot.state)

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
            self.solver.set(j, "yref", np.array([pos[0, j], pos[1, j], psi[j], v[j], w[j]]))
        self.solver.set(self.N, "yref", np.array([pos[0, self.N], pos[1, self.N], psi[self.N]]))
        self.k += 1

        # Solve the optimization problem
        action = self.solver.solve_for_x0(robot.state.values)
        print(action)
        action = DifferentialDriveAction(v = action[0], w = action[1])
        robot.input = action
        
        next_state = self.integrator.simulate(robot.state.values, action.values)
        error = np.linalg.norm(next_state[:2] - pos[:, 0])
        next_state = robot.__class__.create_state(*next_state)
        robot.state = next_state
        
        print(next_state)
        print("Error: ", error)
        print("")
        
        return action, next_state, pos, error

    # def _init_horizon(self, state: DifferentialDriveState):
    #     # initial state
    #     state = state.values.squeeze()
    #     self.opti.set_value(self.state0, state)

    #     # initializing state and action prediction
    #     self.opti.set_initial(self.action, self.action_prediction)
    #     self.opti.set_initial(self.state, self.state_prediction)

    # def _unpack_state(self, state: np.ndarray):
    #     x = state[self.robot.state.index("x")]
    #     y = state[self.robot.state.index("y")]
    #     psi = state[self.robot.state.index("psi")]
    #     return x, y, psi

    # def _unpack_action(self, action: np.ndarray):
    #     v = action[self.robot.input.index("v")]
    #     w = action[self.robot.input.index("w")]
    #     return v, w
