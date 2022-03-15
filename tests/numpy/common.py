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


def _test_odot_wedge(G):
    X = G.random()
    a = G.Log(X)
    b = np.random.normal(0, 1, (X.shape[0], 1))

    test1 = np.dot(G.wedge(a), b)
    test2 = np.dot(G.odot(b), a)
    assert np.allclose(test1, test2)


def _test_left_jacobian_inverse(G):
    X = G.random()
    xi = G.Log(X)
    J_left = G.left_jacobian(xi)
    J_left_inv = G.left_jacobian_inv(xi)

    assert np.allclose(J_left_inv, np.linalg.inv(J_left))

def _test_left_jacobian_numerically(G):
    x_bar = G.Log(G.random())
    J_left = G.left_jacobian(x_bar)

    exp_inv = G.inverse(G.Exp(x_bar))
    J_fd = np.zeros((G.dof, G.dof)) 
    h = 1e-8
    for i in range(G.dof):
        dx = np.zeros((G.dof,1))
        dx[i] = h
        J_fd[:,i] = (G.Log(np.dot(G.Exp(x_bar + dx), exp_inv))/h).flatten()

    
    assert np.allclose(J_fd, J_left)
