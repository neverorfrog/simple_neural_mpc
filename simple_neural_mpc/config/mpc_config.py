from dataclasses import dataclass, field


@dataclass
class CostWeights:
    ex: float = 10
    ey: float = 10
    v: float = 1
    w: float = 1


@dataclass
class Constraints:
    x_min: float = -5
    x_max: float = 5
    y_min: float = -5
    y_max: float = 5
    psi_min: float = -3.14
    psi_max: float = 3.14
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
    neural: bool = False
    cost_weights: CostWeights = field(default_factory=CostWeights)
    constraints: Constraints = field(default_factory=Constraints)
