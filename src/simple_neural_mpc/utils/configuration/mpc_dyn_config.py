from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class CostWeights:
    ex: float = MISSING
    ey: float = MISSING
    F_l: float = MISSING
    F_r: float = MISSING


@dataclass
class InputConstraints:
    F_l_min: float = MISSING
    F_l_max: float = MISSING
    F_r_min: float = MISSING
    F_r_max: float = MISSING


@dataclass
class StateConstraints:
    x_min: float = MISSING
    x_max: float = MISSING
    y_min: float = MISSING
    y_max: float = MISSING
    psi_min: float = MISSING
    psi_max: float = MISSING


@dataclass
class ModelPredictiveControllerConfig:
    dt: float = MISSING
    horizon: int = MISSING
    color: str = MISSING
    cost_weights: CostWeights = field(default_factory=CostWeights)
    input_constraints: InputConstraints = field(default_factory=InputConstraints)
    state_constraints: StateConstraints = field(default_factory=StateConstraints)
