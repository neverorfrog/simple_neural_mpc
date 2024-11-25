from simple_neural_mpc.utils import load_config, project_root
from simple_neural_mpc.utils.configuration import (
    UnicycleConfig,
    DynModelPredictiveControllerConfig,
)

from simple_neural_mpc.utils.trajectory import Circle

from simple_neural_mpc.models.unicycle_dyn.unicycle_dyn_casadi import Unicycle

from simple_neural_mpc.controllers.mpc.mpc_dyn_casadi import ModelPredictiveController as CasadiMPC

from simple_neural_mpc.simulation.trajectory_tracking import TrajectoryTrackingSimulation


if __name__ == "__main__":
    reference = Circle(freq=0.2)
    root = project_root()

    # Bicycle model and corresponding controller
    robot_config = load_config(
        f"{root}/config/models/unicycle.yaml",
        UnicycleConfig,
    )

    controller_config = load_config(
        f"{root}/config/controllers/mpc_dyn.yaml",
        DynModelPredictiveControllerConfig,
    )

    print(controller_config)

    robot = Unicycle(config=robot_config)
    controller = CasadiMPC(robot, controller_config)

    # Simulation
    simulation = TrajectoryTrackingSimulation("mpc_dyn_casadi", robot, controller, reference)
    simulation.run(N=500, animate=True, save=False)
