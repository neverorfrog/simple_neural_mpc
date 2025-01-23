from dataclasses import dataclass, field

import numpy as np


@dataclass
class CostWeights:
    ex: float = 15
    ey: float = 15
    v: float = 0.1
    w: float = 0.1


@dataclass
class Constraints:
    x_min: float = -2
    x_max: float = 2
    y_min: float = -2
    y_max: float = 2
    psi_min: float = -np.pi
    psi_max: float = np.pi
    v_min: float = -1
    v_max: float = 1
    w_min: float = -0.5
    w_max: float = 0.5


@dataclass
class MPCConfig:
    dt: float = 0.01
    horizon: int = 100
    color: str = "red"
    model_name: str = "unicycle"
    is_neural: bool = True
    is_pinn: bool = True
    cost_weights: CostWeights = field(default_factory=CostWeights)
    constraints: Constraints = field(default_factory=Constraints)
