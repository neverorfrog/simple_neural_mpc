import numpy as np

from simple_neural_mpc.controllers import DFBL
from simple_neural_mpc.robots import Unicycle
from simple_neural_mpc.simulation.trajectory import Circle
from simple_neural_mpc.simulation.trajectory_tracking import (
    TrajectoryTrackingSimulation,
)
from simple_neural_mpc.utils import load_config, project_root

if __name__ == "__main__":
    reference = Circle(freq=0.1)
    root = project_root()

    # Bicycle model and corresponding controller
    robot = Unicycle()
    robot.input.v = 0.1
    controller = DFBL(kp=np.array([5, 5]), kd=np.array([2, 2]))

    # Simulation
    simulation = TrajectoryTrackingSimulation("fbl", robot, controller, reference)
    simulation.run(N=500, animate=True, save=False)
