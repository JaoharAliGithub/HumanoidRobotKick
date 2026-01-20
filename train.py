"""
Training entrypoint (logic-only).

This repository contains the task logic (reward, termination, observation builder)
for a humanoid kick task intended for Isaac Lab.

The full Isaac Sim environment/assets are not included here due to size/licensing.
To train in Isaac Lab, wire these functions into your Env/Task class and PPO runner.

This script runs a small sanity check so the repo isn't "empty".
"""

from __future__ import annotations
import numpy as np

from humanoid_kick.reward import compute_reward, RewardWeights, RewardParams
from humanoid_kick.obs import build_observation, ObsSpec
from humanoid_kick.termination import is_fallen, is_timeout, TerminationParams


def sanity_check(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    # Dummy state vectors (sizes are illustrative; adapt to your humanoid)
    q = rng.normal(size=(10,)).astype(np.float32)
    qd = rng.normal(size=(10,)).astype(np.float32)
    base_lin_vel = rng.normal(size=(3,)).astype(np.float32)
    base_ang_vel = rng.normal(size=(3,)).astype(np.float32)
    projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    ball_pos_rel = rng.normal(size=(3,)).astype(np.float32)
    ball_vel_prev = rng.normal(size=(3,)).astype(np.float32)
    ball_vel = ball_vel_prev + rng.normal(scale=0.1, size=(3,)).astype(np.float32)

    foot_rel_to_ball = rng.normal(size=(3,)).astype(np.float32)
    dist_foot_to_ball = float(np.linalg.norm(foot_rel_to_ball))

    # Reward call
    total, terms, kicked_next = compute_reward(
        m_ball=0.45,
        v_ball=ball_vel,
        v_ball_prev=ball_vel_prev,
        up_dot=0.85,
        base_ang_vel=base_ang_vel,
        action=rng.normal(size=(16,)).astype(np.float32),
        dist_kick_foot_to_ball=dist_foot_to_ball,
        alive=True,
        foot_ball_contact=True,
        kicked=False,
        weights=RewardWeights(),
        params=RewardParams(),
    )

    # Observation
    obs = build_observation(
        q=q,
        qd=qd,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        projected_gravity=projected_gravity,
        ball_pos_rel=ball_pos_rel,
        ball_vel=ball_vel,
        foot_pos_rel_to_ball=foot_rel_to_ball,
        spec=ObsSpec(),
    )

    # Termination checks
    term_params = TerminationParams()
    fallen = is_fallen(base_height=0.9, up_dot=0.85, params=term_params)
    timeout = is_timeout(step=10, params=term_params)

    print("Sanity check OK")
    print("obs_dim:", obs.shape[0])
    print("reward_total:", float(total))
    print("kicked_next:", kicked_next)
    print("fallen:", fallen, "timeout:", timeout)
    print("terms:", {k: float(v) for k, v in terms.items()})


def main() -> None:
    sanity_check()


if __name__ == "__main__":
    main()
