import os

import torch
from torch import nn

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
    
class PretrainedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 256)  # 5 input nodes: x, y, theta, v, omega
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 3)  # 3 output nodes: x_next, y_next, theta_next
        
    def forward(self, x):
        x = x.T  #! input has to be a column vector
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = x.T  #! output has to be a column vector
        return x
    
# Define the dynamic system using the trained model and l4casADi
root = project_root()
torch_model = PretrainedModel()
torch_model.load_state_dict(torch.load(f"{root}/simple_neural_mpc/scripts/unicycle_model_state_simple.pth"))
torch_model.eval()

if __name__ == "__main__":
    reference = Circle(freq=0.2)

    # Bicycle model and corresponding controller
    robot_config = load_config(
        f"{root}/config/models/unicycle.yaml",
        UnicycleConfig,
    )

    robot = Unicycle(config=robot_config, neural_network=torch_model)

    mpc_config = load_config(
        f"{root}/config/controllers/mpc_kin.yaml",
        ModelPredictiveControllerConfig,
    )

    controller = ModelPredictiveController(robot, mpc_config)

    # Simulation
    simulation = TrajectoryTrackingSimulation("neural_mpc", robot, controller, reference)
    simulation.run(N=500, animate=True, save=False)
