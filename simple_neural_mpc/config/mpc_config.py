from dataclasses import dataclass, field

import numpy as np


@dataclass
class CostWeights:
    ex: float = 2
    ey: float = 2
    epsi: float = 2
    v: float = 0.5
    w: float = 0.1
    ex_term: float = 5
    ey_term: float = 5
    epsi_term: float = 5


@dataclass
class Constraints:
    x_min: float = -5
    x_max: float = 5
    y_min: float = -5
    y_max: float = 5
    psi_min: float = -10
    psi_max: float = 10
    v_min: float = -5
    v_max: float = 5
    w_min: float = -4
    w_max: float = 4


@dataclass
class MPCConfig:
    dt: float = 0.05
    horizon: int = 100
    color: str = "red"
    model_name: str = "unicycle"
    is_neural: bool = False
    predicts_state: bool = False
    is_pinn: bool = False
    cost_weights: CostWeights = field(default_factory=CostWeights)
    constraints: Constraints = field(default_factory=Constraints)
