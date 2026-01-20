import numpy as np
from humanoid_kick.reward import compute_reward

def test_power_reward_requires_kick_or_post_kick_gate():
    m = 1.0
    base_ang = np.zeros(3)
    action = np.zeros(8)

    # No contact, ball gaining speed (should not count pre-kick)
    total, terms, kicked = compute_reward(
        m_ball=m,
        v_ball=np.array([2.0, 0.0, 0.0]),
        v_ball_prev=np.array([0.0, 0.0, 0.0]),
        up_dot=1.0,
        base_ang_vel=base_ang,
        action=action,
        dist_kick_foot_to_ball=0.3,
        alive=True,
        foot_ball_contact=False,
        kicked=False,
    )
    assert terms["kicked"] == 0.0
    assert terms["r_power"] == 0.0

def test_uprightness_clamps():
    m = 1.0
    base_ang = np.zeros(3)
    action = np.zeros(8)

    total, terms, kicked = compute_reward(
        m_ball=m,
        v_ball=np.zeros(3),
        v_ball_prev=np.zeros(3),
        up_dot=0.0,
        base_ang_vel=base_ang,
        action=action,
        dist_kick_foot_to_ball=1.0,
        alive=True,
        foot_ball_contact=False,
        kicked=False,
    )
    assert terms["r_upright"] == 0.0
