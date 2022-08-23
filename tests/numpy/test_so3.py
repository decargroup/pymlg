from pylie import SO3 as G
import numpy as np
import common


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


def test_odot_wedge():
    common._test_odot_wedge(G)


def test_group_jacobians():
    common._test_left_jacobian_inverse(G)


def test_left_jacobian_numerically():
    common._test_left_jacobian_numerically(G)


def test_adjoint_identity():
    common._test_adjoint_identity(G)


def test_from_euler():
    theta = np.array([0.1, 0.2, 0.3])
    C = G.from_euler(theta)
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(np.dot(C, np.transpose(C)), np.identity(3))

    C = G.from_euler(theta, order=[1, 2, 3])
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(np.dot(C, np.transpose(C)), np.identity(3))

    C = G.from_euler(theta, order=[3, 1, 3])
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(np.dot(C, np.transpose(C)), np.identity(3))


def test_quaternion():
    q = np.array([1, 2, 3, 4]).reshape((-1, 1))
    q = q / np.linalg.norm(q)
    C = G.from_quat(q, order="wxyz")
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(np.dot(C, np.transpose(C)), np.identity(3))

    q_test = G.to_quat(C, order="wxyz")
    assert np.allclose(q, q_test)

    C_test = G.from_quat(-q, order="wxyz")
    assert np.allclose(C, C_test)

    # Different order
    C = G.from_quat(q, order="xyzw")
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(np.dot(C, np.transpose(C)), np.identity(3))

    q_test = G.to_quat(C, order="xyzw")
    assert np.allclose(q, q_test)

    C_test = G.from_quat(-q, order="xyzw")
    assert np.allclose(C, C_test)


if __name__ == "__main__":
    test_quaternion()
