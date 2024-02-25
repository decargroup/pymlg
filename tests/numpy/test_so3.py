"""
Perform any other group-specific tests that are not part of the standard tests.
"""

from pymlg import SO3 as G
import numpy as np

def test_euler():
    theta = np.array([0.1, 0.2, 0.3])
    C = G.from_euler(theta)
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(np.dot(C, np.transpose(C)), np.identity(3))

    C = G.from_euler(theta, order=[1, 2, 3])
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(np.dot(C, np.transpose(C)), np.identity(3))

    theta_test = G.to_euler(C, order="123")
    assert np.allclose(theta, theta_test)

    C = G.from_euler(theta, order=[3, 2, 1])
    assert np.isclose(np.linalg.det(C), 1)
    assert np.allclose(np.dot(C, np.transpose(C)), np.identity(3))

    theta_test = G.to_euler(C, order="321")
    assert np.allclose(theta, theta_test)

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
    test_euler()
    test_quaternion()
