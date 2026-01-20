from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObsSpec:
    use_projected_gravity: bool = True
    use_ball_vel: bool = True
    use_foot_rel: bool = True

def build_observation(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    base_lin_vel: np.ndarray,
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray | None,
    ball_pos_rel: np.ndarray,
    ball_vel: np.ndarray | None,
    foot_pos_rel_to_ball: np.ndarray | None,
    spec: ObsSpec = ObsSpec(),
) -> np.ndarray:
    parts = [q, qd, base_lin_vel, base_ang_vel, ball_pos_rel]
    if spec.use_projected_gravity:
        assert projected_gravity is not None
        parts.append(projected_gravity)
    if spec.use_ball_vel:
        assert ball_vel is not None
        parts.append(ball_vel)
    if spec.use_foot_rel:
        assert foot_pos_rel_to_ball is not None
        parts.append(foot_pos_rel_to_ball)
    return np.concatenate(parts, axis=0).astype(np.float32)
