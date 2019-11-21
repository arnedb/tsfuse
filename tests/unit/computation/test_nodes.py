from tsfuse.transformers import Mean, Add


def test_n_inputs():
    assert Mean().n_inputs == 1
    assert Add().n_inputs == 2
