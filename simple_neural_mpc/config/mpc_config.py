from dataclasses import dataclass, field

import numpy as np


@dataclass
class CostWeights:
    ex: float = 5
    ey: float = 5
    epsi: float = 0.1
    v: float = 0.01
    w: float = 0.01
    ex_term: float = 10
    ey_term: float = 10
    epsi_term: float = 1


@dataclass
class Constraints:
    x_min: float = -1
    x_max: float = 1
    y_min: float = -1
    y_max: float = 1
    psi_min: float = -1
    psi_max: float = 1
    v_min: float = -1
    v_max: float = 1
    w_min: float = -1
    w_max: float = 1


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
