import numpy as np
import sys
sys.path.append('/home/flavio/Scrivania/simple_neural_mpc')
print(sys.path)
from simple_neural_mpc.controllers import DFBL, FBL
from simple_neural_mpc.models.differential_drive_kin import DifferentialDrive
from simple_neural_mpc.simulation.trajectory_tracking import (
    TrajectoryTrackingSimulation,
)
from simple_neural_mpc.utils import load_config
from simple_neural_mpc.utils.configuration import DifferentialDriveConfig
from simple_neural_mpc.utils.trajectory import Circle

if __name__ == "__main__":
    reference = Circle()

    # Bicycle model and corresponding controller
    robot_config = load_config(
        f"config/models/differential_drive.yaml",
        DifferentialDriveConfig,
    )
    robot = DifferentialDrive(config=robot_config)
    robot.input.v = 0.1
    # controller = FBL(kp=np.array([1,1]),kd=np.array([1,1]))
    controller = DFBL(kp=np.array([5, 5]), kd=np.array([2, 2]))

    # Simulation
    simulation = TrajectoryTrackingSimulation(
        "boh", robot, controller, reference
    )
    simulation.run(N=50, animate=True, save=False)
