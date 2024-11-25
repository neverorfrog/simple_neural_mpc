from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class Car:
    m: float = MISSING
    l: float = MISSING
    
@dataclass
class DifferentialDriveConfig:
    dt: float = MISSING
    car: Car = field(default_factory=Car)