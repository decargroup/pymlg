import numpy as np
from scipy.linalg import expm, logm
from pylie import MatrixLieGroup, SO2, SO3, SE2, SE3, SE23, SL3
import pylie
import pytest


@pytest.mark.parametrize("G", [SO2, SO3, SE2, SE3, SE23, SL3])
class TestStandardNumpy:
    def test_wedge_vee(self, G: MatrixLieGroup):
        x = 0.1 * np.random.random((G.dof, 1))
        x_test = G.vee(G.wedge(x))
        if G.dof > 1:
            assert x_test.shape == (G.dof, 1)
        assert np.allclose(x, x_test, 1e-15)

    def test_exp(self, G: MatrixLieGroup):
        x = np.random.random((G.dof, 1))
        Xi = G.wedge(x)
        X = G.exp(Xi)
        X_test = expm(Xi)
        assert np.allclose(X, X_test)

    def test_log(self, G: MatrixLieGroup):
        X = G.random()
        Xi = G.log(X)
        Xi_test = logm(X)
        assert np.allclose(Xi, Xi_test)

    def test_exp_log_inverse(self, G: MatrixLieGroup):
        X = G.random()
        Xi = G.log(X)
        assert np.allclose(X, G.exp(G.log(X)))
        assert np.allclose(Xi, G.log(G.exp(Xi)))

    def test_capital_exp_log_inverse(self, G: MatrixLieGroup):
        T = G.random()
        x = G.Log(T)
        assert np.allclose(T, G.Exp(x))

        if G.dof > 1:
            assert x.shape == (G.dof, 1)

    def test_odot_wedge(self, G: MatrixLieGroup):
        X = G.random()
        a = G.Log(X)
        b = np.random.normal(0, 1, (X.shape[0], 1))

        test1 = np.dot(G.wedge(a), b)
        test2 = np.dot(G.odot(b), a)
        assert np.allclose(test1, test2)

    def test_left_jacobian_inverse(self, G: MatrixLieGroup):
        X = G.random()
        xi = G.Log(X)
        J_left = G.left_jacobian(xi)
        J_left_inv = G.left_jacobian_inv(xi)

        assert np.allclose(J_left_inv, np.linalg.inv(J_left))

    def test_right_jacobian_inverse(self, G: MatrixLieGroup):
        X = G.random()
        xi = G.Log(X)
        J_right = G.right_jacobian(xi)
        J_right_inv = G.right_jacobian_inv(xi)

        assert np.allclose(J_right_inv, np.linalg.inv(J_right))

    def test_left_jacobian_numerically(self, G: MatrixLieGroup):
        np.random.seed(0)
        x_bar = G.Log(G.random())
        J_left = G.left_jacobian(x_bar)

        exp_inv = G.inverse(G.Exp(x_bar))
        J_fd = np.zeros((G.dof, G.dof))
        h = 1e-8
        for i in range(G.dof):
            dx = np.zeros((G.dof, 1))
            dx[i] = h
            J_fd[:, i] = (G.Log(np.dot(G.Exp(x_bar + dx), exp_inv)) / h).ravel()

        assert np.allclose(J_fd, J_left, atol=1e-7)

    def test_adjoint_identity(self, G: MatrixLieGroup):
        X = G.random()
        xi = G.Log(G.random())

        side1 = G.wedge(np.dot(G.adjoint(X), xi))
        side2 = np.dot(X, np.dot(G.wedge(xi), G.inverse(X)))
        assert np.allclose(side1, side2)

    def test_inverse(self, G: MatrixLieGroup):
        X = G.random()
        assert np.allclose(G.inverse(G.inverse(X)), X)
        assert np.allclose(G.inverse(X), np.linalg.inv(X))
        assert np.allclose(G.inverse(G.identity()), G.identity())
        assert np.allclose(G.inverse(X) @ X, G.identity())

    def do_tests(self, G: MatrixLieGroup):
        self.test_wedge_vee(G)
        self.test_exp(G)
        self.test_log(G)
        self.test_exp_log_inverse(G)
        self.test_capital_exp_log_inverse(G)
        self.test_odot_wedge(G)
        self.test_left_jacobian_inverse(G)
        self.test_right_jacobian_inverse(G)
        self.test_left_jacobian_numerically(G)
        self.test_adjoint_identity(G)
        self.test_inverse(G)


if __name__ == "__main__":
    # For debugging purposes
    test = TestStandardNumpy()
    test.do_tests(SO2)
    test.do_tests(SO3)
    test.do_tests(SE2)
    test.do_tests(SE3)
    test.do_tests(SE23)
    test.do_tests(SL3)
