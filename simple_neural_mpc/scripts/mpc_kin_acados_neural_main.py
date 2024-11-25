import os

import torch

from simple_neural_mpc.controllers.neural_mpc.mpc_kin_acados_neural import (
    ModelPredictiveController,
)
from simple_neural_mpc.models.unicycle_kin.unicycle_kin_acados_neural import (
    Unicycle,
)
from simple_neural_mpc.simulation.trajectory_tracking import (
    TrajectoryTrackingSimulation,
)
from simple_neural_mpc.utils.configuration.mpc_kin_config import (
    ModelPredictiveControllerConfig,
)
from simple_neural_mpc.utils.configuration.unicycle_config import UnicycleConfig
from simple_neural_mpc.utils.misc import load_config, project_root
from simple_neural_mpc.utils.trajectory import Circle

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.input_layer = torch.nn.Linear(3, 64)
        # self.hidden_layer = torch.nn.Linear(64, 64)
        # self.output_layer = torch.nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.T  #! input has to be a column vector
        # x = self.input_layer(x)
        # x = torch.relu(x)
        # x = self.hidden_layer(x)
        # x = torch.relu(x)
        # x = self.output_layer(x)
        # x = x.T  #! output has to be a column vector
        return x


if __name__ == "__main__":
    reference = Circle(freq=0.2)
    root = project_root()

    # Bicycle model and corresponding controller
    robot_config = load_config(
        f"{root}/config/models/unicycle.yaml",
        UnicycleConfig,
    )

    robot = Unicycle(config=robot_config, neural_network=PyTorchModel())

    mpc_config = load_config(
        f"{root}/config/controllers/mpc_kin.yaml",
        ModelPredictiveControllerConfig,
    )

    controller = ModelPredictiveController(robot, mpc_config)

    # Simulation
    simulation = TrajectoryTrackingSimulation("neural_mpc", robot, controller, reference)
    simulation.run(N=100, animate=True, save=False)
