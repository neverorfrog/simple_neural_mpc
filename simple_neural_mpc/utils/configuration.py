from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
@dataclass
class CostWeights:
    ex: float = MISSING
    ey: float = MISSING
    v: float = MISSING
    w: float = MISSING


@dataclass
class InputConstraints:
    v_min: float = MISSING
    v_max: float = MISSING
    w_min: float = MISSING
    w_max: float = MISSING


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
    input_constraints: InputConstraints = field(
        default_factory=InputConstraints
    )
    state_constraints: StateConstraints = field(
        default_factory=StateConstraints
    )


@dataclass
class DifferentialDriveConfig:
    dt: float = MISSING
