from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TerminationParams:
    min_base_height: float = 0.75
    min_up_dot: float = 0.55
    max_steps: int = 600

def is_fallen(base_height: float, up_dot: float, params: TerminationParams) -> bool:
    return (base_height < params.min_base_height) or (up_dot < params.min_up_dot)

def is_timeout(step: int, params: TerminationParams) -> bool:
    return step >= params.max_steps
