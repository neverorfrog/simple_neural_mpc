from simple_neural_mpc.controllers import MPC
from simple_neural_mpc.robots import Unicycle
from simple_neural_mpc.simulation.trajectory import Circle
from simple_neural_mpc.simulation.trajectory_tracking import (
    TrajectoryTrackingSimulation,
)
from simple_neural_mpc.utils import load_config, project_root

from simple_neural_mpc.neural_modeling.learner.mlp import MLP
import torch
from simple_neural_mpc.robots.unicycle import UnicycleState

pinn = MLP(3, 2)
pinn.load_state_dict(
    torch.load(
        f"{project_root()}/simple_neural_mpc/neural_modeling/models/unicycle_80_epochs.pth",
        weights_only=True,
        map_location=torch.device("cpu"),
    ),
    strict=False,
)
pinn.eval()

robot = Unicycle(pinn)
robot.state = UnicycleState(0, 0, 0) 
controller = MPC(robot, to_generate=True)

reference = Circle(freq=0.1)
simulation = TrajectoryTrackingSimulation("mpc", robot, controller, reference)
simulation.run(N=300, animate=True, save=False)