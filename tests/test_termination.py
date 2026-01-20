from humanoid_kick.termination import is_fallen, TerminationParams

def test_fall_by_height():
    params = TerminationParams(min_base_height=0.75)
    assert is_fallen(base_height=0.5, up_dot=1.0, params=params)

def test_fall_by_orientation():
    params = TerminationParams(min_up_dot=0.6)
    assert is_fallen(base_height=1.0, up_dot=0.2, params=params)

def test_not_fallen():
    params = TerminationParams()
    assert not is_fallen(base_height=1.0, up_dot=1.0, params=params)
