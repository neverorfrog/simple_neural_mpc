from simple_neural_mpc.controllers.mpc.mpc_kin_acados import (
    ModelPredictiveController as AcadosMPC,
)
from simple_neural_mpc.models.unicycle_kin.unicycle_kin_acados import Unicycle
from simple_neural_mpc.simulation.trajectory_tracking import (
    TrajectoryTrackingSimulation,
)
from simple_neural_mpc.utils import load_config, project_root
from simple_neural_mpc.utils.configuration import (
    KinModelPredictiveControllerConfig,
    UnicycleConfig,
)
from simple_neural_mpc.utils.trajectory import Circle

if __name__ == "__main__":
    reference = Circle(freq=0.2)
    root = project_root()

    # Bicycle model and corresponding controller
    robot_config = load_config(
        f"{root}/config/models/unicycle.yaml",
        UnicycleConfig,
    )

    controller_config = load_config(
        f"{root}/config/controllers/mpc_kin.yaml",
        KinModelPredictiveControllerConfig,
    )

    print(controller_config)

    robot = Unicycle(config=robot_config)
    controller = AcadosMPC(robot, controller_config)

    # Simulation
    simulation = TrajectoryTrackingSimulation(
        "mpc_kin_acados", robot, controller, reference
    )
    simulation.run(N=1, animate=False, save=False)
