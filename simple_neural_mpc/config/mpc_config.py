from dataclasses import dataclass, field


@dataclass
class CostWeights:
    ex: float = 0.1
    ey: float = 0.1
    v: float = 0.01
    w: float = 0.01


@dataclass
class InputConstraints:
    v_min: float = 0.1
    v_max: float = 0.1
    w_min: float = 0.1
    w_max: float = 0.1


@dataclass
class StateConstraints:
    x_min: float = -10
    x_max: float = 10
    y_min: float = -10
    y_max: float = 10
    psi_min: float = -1
    psi_max: float = 1


@dataclass
class MPCConfig:
    dt: float = 0.01
    horizon: int = 100
    color: str = "red"
    model_name: str = "unicycle"
    neural: bool = False
    cost_weights: CostWeights = field(default_factory=CostWeights)
    input_constraints: InputConstraints = field(default_factory=InputConstraints)
    state_constraints: StateConstraints = field(default_factory=StateConstraints)
