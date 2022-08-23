from pylie import SO2 as G
import common
import numpy as np

def test_wedge_vee():
    common._test_wedge_vee(G)


def test_exp():
    common._test_exp(G)


def test_log():
    common._test_log(G)


def test_exp_log_inverse():
    common._test_exp_log_inverse(G)


def test_capital_exp_log_inverse():
    common._test_capital_exp_log_inverse(G)


def test_group_jacobians():
    common._test_left_jacobian_inverse(G)


def test_adjoint_identity():
    X = G.random()
    xi = G.Log(G.random())

    side1 = G.wedge(xi)
    side2 = np.dot(X, np.dot(G.wedge(xi), G.inverse(X)))
    assert np.allclose(side1, side2)


def test_odot_wedge():
    common._test_odot_wedge(G)


if __name__ == "__main__":
    test_adjoint_identity()
