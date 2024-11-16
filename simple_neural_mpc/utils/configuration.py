from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class CostWeights:
    ex: float
    ey: float


@dataclass
class InputConstraints:
    v_min: float
    v_max: float
    w_min: float
    w_max: float


@dataclass
class StateConstraints:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    psi_min: float
    psi_max: float


@dataclass
class ModelPredictiveControllerConfig:
    horizon: int = MISSING
    color: str = MISSING
    cost_weights: CostWeights = field(default_factory=CostWeights)
    input_constraints: InputConstraints = field(
        default_factory=InputConstraints
    )
    state_constraints: StateConstraints = field(
        default_factory=StateConstraints
    )


@dataclass
class DifferentialDriveConfig:
    dt: float = MISSING
