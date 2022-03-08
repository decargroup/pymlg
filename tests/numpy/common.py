import numpy as np
from scipy.linalg import expm, logm


def _test_wedge_vee(G):
    x = 0.1 * np.random.random((G.dof, 1))
    x_test = G.vee(G.wedge(x))
    assert np.allclose(x, x_test, 1e-15)


def _test_exp(G):
    x = np.random.random((G.dof, 1))
    Xi = G.wedge(x)
    X = G.exp(Xi)
    X_test = expm(Xi)
    assert np.allclose(X, X_test)


def _test_log(G):
    X = G.random()
    Xi = G.log(X)
    Xi_test = logm(X)
    assert np.allclose(Xi, Xi_test)


def _test_exp_log_inverse(G):
    X = G.random()
    Xi = G.log(X)
    assert np.allclose(X, G.exp(G.log(X)))
    assert np.allclose(Xi, G.log(G.exp(Xi)))
