from controllers import DFBL, FBL
from models.differential_drive import DifferentialDrive
from simulation.trajectory_tracking import TrajectoryTrackingSimulation
from utils.common_utils import *
from utils.trajectory import Circle

if __name__ == "__main__":
    reference = Circle()

    # Bicycle model and corresponding controller
    robot_config = load_config(
        f"/home/neverorfrog/code/vehicle-control/config/models/differential_drive.yaml"
    )
    robot = DifferentialDrive(config=robot_config)
    robot.input.v = 0.1
    # controller = FBL(kp=np.array([1,1]),kd=np.array([1,1]))
    controller = DFBL(kp=np.array([5, 5]), kd=np.array([2, 2]))

    # Simulation
    simulation = TrajectoryTrackingSimulation(
        "boh", robot, controller, reference
    )
    simulation.run(N=50, animate=True, save=True)
