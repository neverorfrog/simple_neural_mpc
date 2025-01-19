import importlib
import sys
from pathlib import Path

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from simple_neural_mpc.config.mpc_config import MPCConfig as config
from simple_neural_mpc.controllers.controller import Controller
from simple_neural_mpc.robots.unicycle import Unicycle, UnicycleAction
from simple_neural_mpc.simulation.trajectory import Trajectory

np.random.seed(31)


class MPC(Controller):
    def __init__(
        self,
        robot: Unicycle,
        to_generate: bool = False,
    ):
        """Optimizer Initialization"""
        self.robot = robot
        self.N = config.horizon
        self.acados_gen_path = Path("acados_generated_files")
        self.to_generate = to_generate
        self.k = 0  # current iteration
        self._init_ocp()
        self.ode = lambda x, u: np.array([u[0] * np.cos(x[2]), u[1] * np.sin(x[2]), u[1]])

    def _init_ocp(self):
        if not self.to_generate:
            try:
                if self.acados_gen_path.is_dir():
                    sys.path.append(str(self.acados_gen_path))
                acados_ocp_solver_pyx = importlib.import_module(
                    "c_generated_code.acados_ocp_solver_pyx"
                )
                self.solver: AcadosOcpSolver = (
                    acados_ocp_solver_pyx.AcadosOcpSolverCython(
                        self.robot.model.name, "SQP", self.N
                    )
                )
                print("Acados cython module imported successfully.")
            except ImportError:
                print("Acados cython code was not found. Generating it now...")
                self.to_generate = True
        if self.to_generate:
            # OCP
            self.ocp = AcadosOcp()
            self.ocp.model = self.robot.model
            self.ocp.code_export_directory = self.acados_gen_path / ("c_generated_code")
            json_file = str(self.acados_gen_path / ("acados_ocp.json"))

            # Dimensions
            self.ns = self.ocp.model.x.rows()
            self.na = self.ocp.model.u.rows()

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
                np.array([10, 10, 0.1, 1, 1])
            )  # weight matrix for stage cost
            self.ocp.cost.W_e = np.diag(
                np.array([15, 15, 1])
            )  # weight matrix for terminal cost

            # Constraints
            x_0, y_0, psi_0 = self.robot.state.values
            self.ocp.constraints.x0 = np.array([x_0, y_0, psi_0])

            # Reference
            self.ocp.cost.yref = np.zeros((self.n_opt))
            self.ocp.cost.yref_e = np.zeros((self.n_opt_e))

            # Solver Options
            self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
            self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
            self.ocp.solver_options.integrator_type = "ERK"
            self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
            self.ocp.solver_options.tol = 1e-3
            self.ocp.solver_options.qp_tol = 1e-3
            self.ocp.solver_options.nlp_solver_max_iter = 500
            self.ocp.solver_options.qp_solver_iter_max = 100
            self.ocp.solver_options.print_level = 0

            # Prediction Horizon
            self.ocp.solver_options.tf = config.horizon * config.dt
            self.ocp.solver_options.N_horizon = config.horizon

            # L4Casadi Stuff
            if config.neural is True and self.robot.neural_network is not None:
                self.ocp.solver_options.model_external_shared_lib_dir = (
                    self.robot.l4casadi_model.shared_lib_dir
                )
                self.ocp.solver_options.model_external_shared_lib_name = (
                    self.robot.l4casadi_model.name
                )

            # Debug Stuff
            self.ocp.solver_options.print_level = 0

            # Generate c code
            AcadosOcpSolver.generate(self.ocp, json_file=json_file)
            AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
            if self.acados_gen_path.is_dir():
                sys.path.append(str(self.acados_gen_path))
            acados_ocp_solver_pyx = importlib.import_module(
                "c_generated_code.acados_ocp_solver_pyx"
            )
            self.solver: AcadosOcpSolver = acados_ocp_solver_pyx.AcadosOcpSolverCython(
                self.robot.model.name, self.ocp.solver_options.nlp_solver_type, self.N
            )

    def command(self, robot: Unicycle, reference: Trajectory, t: float):
        # generate trajectory for next N steps
        t = np.linspace(
            self.k * config.dt,
            (self.k + self.N) * config.dt,
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

        cur_state = robot.state.values
        # cur_action = np.array([action.v, action.w])
        next_state = cur_state + 0.1 * np.array(
            [action.v * np.cos(cur_state[2]), action.v * np.sin(cur_state[2]), action.w]
        )
        # next_state = self.robot.integrate(cur_state, cur_action, self.ode, config.dt)
        # print(next_state)

        next_state = self.solver.get(1, "x")

        error = np.linalg.norm(next_state[:2] - pos[:, 0])
        next_state = robot.__class__.create_state(*next_state)
        robot.state = next_state

        return action, next_state, pos, error
