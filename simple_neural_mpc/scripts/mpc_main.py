import numpy as np

from simple_neural_mpc.controllers import DFBL
from simple_neural_mpc.controllers.mpc import ModelPredictiveController
from simple_neural_mpc.models.differential_drive import DifferentialDrive
from simple_neural_mpc.simulation.trajectory_tracking import (
    TrajectoryTrackingSimulation,
)
from simple_neural_mpc.utils import load_config
from simple_neural_mpc.utils.configuration import (
    DifferentialDriveConfig,
    ModelPredictiveControllerConfig,
)
from simple_neural_mpc.utils.trajectory import Circle

if __name__ == "__main__":
    reference = Circle(freq=0.2)

    # Bicycle model and corresponding controller
    robot_config = load_config(
        f"/home/neverorfrog/code/trajectory-tracking-neural-network/config/models/differential_drive.yaml",
        DifferentialDriveConfig,
    )

    controller_config = load_config(
        f"/home/neverorfrog/code/trajectory-tracking-neural-network/config/controllers/mpc.yaml",
        ModelPredictiveControllerConfig,
    )

    print(controller_config)

    robot = DifferentialDrive(config=robot_config)
    controller = ModelPredictiveController(robot, controller_config)

    # Simulation
    simulation = TrajectoryTrackingSimulation(
        "boh", robot, controller, reference
    )
    simulation.run(N=500, animate=True, save=False)
