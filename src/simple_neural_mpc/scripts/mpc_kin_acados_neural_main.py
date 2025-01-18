import os

import torch

from neural_model_identification.learner.nn.mlp import MLP_Pinn, MLP
from neural_model_identification.parameters.train_params import TrainParams
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
from simple_neural_mpc.utils.trajectory import Circle, Ellipse, Eight, Zero, Dritto

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Define the dynamic system using the trained model and l4casADi
root = project_root()

torch_model = MLP_Pinn(TrainParams.state_dim, TrainParams.input_dim, 124)
torch_model.load_state_dict(
    torch.load(
        f"{root}/src/neural_model_identification/model_epochs_80_345_extra_small_input_maybe_to_many_ep.pth",
        weights_only=True,
        map_location=torch.device("cpu"),
    ),
    strict=False,
)

# torch_model = MLP(
#     TrainParams.state_dim, TrainParams.input_dim, TrainParams.latent_dim, is_in_mpc=True
# )
# torch_model.load_state_dict(
#     torch.load(
#         f"{root}/src/neural_model_identification/trained_models/kin_unicycle/model.pth",
#         weights_only=True,
#         map_location=torch.device("cpu"),
#     ),
#     strict=False,
# )
# torch_model.eval()

if __name__ == "__main__":
    reference = Circle(freq=0.3)
    
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

    controller = ModelPredictiveController(robot, mpc_config, to_generate=True)

    # Simulation
    simulation = TrajectoryTrackingSimulation("neural_mpc", robot, controller, reference)
    simulation.run(N=500, animate=True, save=True)
