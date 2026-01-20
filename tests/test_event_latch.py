import numpy as np
from humanoid_kick.reward import compute_reward

def test_kick_latch_only_triggers_once():
    m = 1.0
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([2.0, 0.0, 0.0])
    base_ang = np.zeros(3)
    action = np.zeros(8)

    kicked = False

    # first contact: should latch
    total, terms, kicked = compute_reward(
        m_ball=m,
        v_ball=v1,
        v_ball_prev=v0,
        up_dot=1.0,
        base_ang_vel=base_ang,
        action=action,
        dist_kick_foot_to_ball=0.1,
        alive=True,
        foot_ball_contact=True,
        kicked=kicked,
    )
    assert terms["kicked_now"] == 1.0
    assert kicked is True

    # second contact: should not re-trigger kicked_now
    total2, terms2, kicked2 = compute_reward(
        m_ball=m,
        v_ball=v1,
        v_ball_prev=v1,
        up_dot=1.0,
        base_ang_vel=base_ang,
        action=action,
        dist_kick_foot_to_ball=0.1,
        alive=True,
        foot_ball_contact=True,
        kicked=kicked,
    )
    assert terms2["kicked_now"] == 0.0
    assert kicked2 is True
