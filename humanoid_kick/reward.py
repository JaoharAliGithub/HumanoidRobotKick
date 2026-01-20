from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class RewardWeights:
    w_power: float = 10.0
    w_alive: float = 1.0
    w_upright: float = 1.0
    w_approach: float = 0.2
    w_ang_vel: float = 0.05
    w_action: float = 0.01

@dataclass(frozen=True)
class RewardParams:
    # Power shaping
    energy_log_k: float = 1.0
    v_min_contact: float = 0.5  # minimum ball speed to treat contact as "real kick"
    # Approach shaping
    approach_alpha: float = 6.0  # exp(-alpha * dist)
    # Upright shaping
    upright_min: float = 0.6  # below this, uprightness reward becomes 0

def ball_kinetic_energy(m_ball: float, v_ball: np.ndarray) -> float:
    v2 = float(np.dot(v_ball, v_ball))
    return 0.5 * m_ball * v2

def uprightness_from_up_dot(up_dot: float, upright_min: float) -> float:
    # maps up_dot in [upright_min, 1] -> [0, 1]
    if up_dot <= upright_min:
        return 0.0
    return min(1.0, (up_dot - upright_min) / (1.0 - upright_min))

def approach_reward(dist_foot_to_ball: float, alpha: float) -> float:
    # bounded (0,1], larger when closer
    return math.exp(-alpha * dist_foot_to_ball)

def power_reward_from_energy_delta(delta_E: float, log_k: float) -> float:
    # stable against spikes
    if delta_E <= 0.0:
        return 0.0
    return math.log1p(log_k * delta_E)

def compute_reward(
    *,
    m_ball: float,
    v_ball: np.ndarray,
    v_ball_prev: np.ndarray,
    up_dot: float,
    base_ang_vel: np.ndarray,
    action: np.ndarray,
    dist_kick_foot_to_ball: float,
    alive: bool,
    foot_ball_contact: bool,
    kicked: bool,  # latch: whether a kick has already occurred in this episode
    weights: RewardWeights = RewardWeights(),
    params: RewardParams = RewardParams(),
) -> tuple[float, dict[str, float], bool]:
    """
    Returns:
      total_reward
      reward_terms dict (for logging)
      kicked_next (updated latch)
    """
    # energies
    E = ball_kinetic_energy(m_ball, v_ball)
    E_prev = ball_kinetic_energy(m_ball, v_ball_prev)
    delta_E = max(0.0, E - E_prev)

    # Determine if we latch "kicked" this step
    v_mag = float(np.linalg.norm(v_ball))
    kicked_now = (not kicked) and foot_ball_contact and (v_mag >= params.v_min_contact)
    kicked_next = kicked or kicked_now

    # Terms
    r_alive = 1.0 if alive else 0.0
    r_upright = uprightness_from_up_dot(up_dot, params.upright_min)

    # Pre-contact shaping only (optional but nice)
    r_approach = 0.0 if kicked_next else approach_reward(dist_kick_foot_to_ball, params.approach_alpha)

    # Power reward: only meaningful at/after kick. You can choose:
    # - gate strictly to kicked_now, OR
    # - gate to kicked_next (gives a short tail during contact resolution)
    r_power = power_reward_from_energy_delta(delta_E, params.energy_log_k) if kicked_next else 0.0

    # Penalties
    p_ang = float(np.dot(base_ang_vel, base_ang_vel))  # ||omega||^2
    p_act = float(np.dot(action, action))  # ||u||^2

    total = (
        weights.w_power * r_power
        + weights.w_alive * r_alive
        + weights.w_upright * r_upright
        + weights.w_approach * r_approach
        - weights.w_ang_vel * p_ang
        - weights.w_action * p_act
    )

    terms = {
        "r_power": r_power,
        "r_alive": r_alive,
        "r_upright": r_upright,
        "r_approach": r_approach,
        "p_ang": p_ang,
        "p_action": p_act,
        "delta_E": delta_E,
        "kicked_now": 1.0 if kicked_now else 0.0,
        "kicked": 1.0 if kicked_next else 0.0,
    }
    return total, terms, kicked_next
