"""
Perform any other group-specific tests that are not part of the standard tests.
"""

from pymlg import SE3 as G
import numpy as np

def test_odot():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    X = G.odot(p)
    assert X.shape == (4, 6)
    assert np.allclose(G.wedge(x) @ p, X @ x)

def test_ocircle():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    X = G.ocircle(p)
    assert X.shape == (6, 4)
    assert np.allclose(p.T @ G.wedge(x), x.T @ X)

if __name__ == "__main__":
    # test_euler()
    # test_quaternion()
    test_odot()
    test_ocircle()
