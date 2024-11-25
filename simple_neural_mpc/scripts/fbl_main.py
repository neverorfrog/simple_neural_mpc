import numpy as np

from simple_neural_mpc.controllers import DFBL
from simple_neural_mpc.models.unicycle_kin.unicycle_kin_casadi import Unicycle
from simple_neural_mpc.simulation.trajectory_tracking import (
    TrajectoryTrackingSimulation,
)
from simple_neural_mpc.utils import load_config, project_root
from simple_neural_mpc.utils.configuration import UnicycleConfig
from simple_neural_mpc.utils.trajectory import Circle

if __name__ == "__main__":
    reference = Circle()
    root = project_root()

    # Bicycle model and corresponding controller
    robot_config = load_config(
        f"{root}/config/models/unicycle.yaml",
        UnicycleConfig,
    )
    robot = Unicycle(config=robot_config)
    robot.input.v = 0.1
    # controller = FBL(kp=np.array([1,1]),kd=np.array([1,1]))
    controller = DFBL(kp=np.array([5, 5]), kd=np.array([2, 2]))

    # Simulation
    simulation = TrajectoryTrackingSimulation("fbl", robot, controller, reference)
    simulation.run(N=500, animate=True, save=False)
