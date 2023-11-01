import numpy as np
from scipy.linalg import expm, logm
from pymlg import MatrixLieGroup

class StandardTests:
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
        Xi = np.array(Xi).copy()
        X_test = expm(Xi)
        assert np.allclose(X, X_test)

    def test_log(self, G: MatrixLieGroup):
        X = G.random()
        Xi = G.log(X)
        X = np.array(X).copy()
        Xi_test = logm(X)
        assert np.allclose(Xi, Xi_test)

    def test_log_zero(self, G: MatrixLieGroup):
        x = np.zeros((G.dof, 1))
        X = G.Exp(x)
        Xi = G.log(X)
        X = np.array(X).copy()
        Xi_test = logm(X)
        assert np.allclose(Xi, Xi_test)

    def test_capital_log_zero(self, G: MatrixLieGroup):
        x = np.zeros((G.dof, 1))
        X = G.Exp(x)
        x_test = G.Log(X)
        assert np.allclose(x, x_test)


    def test_capital_log_small_value(self, G: MatrixLieGroup):
        x = np.zeros((G.dof, 1))
        x[0] = 1e-8
        X = G.Exp(x)
        x_test = G.Log(X)
        assert not np.isnan(x_test).any()
        assert np.allclose(x, x_test)

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

    
    def test_left_jacobian_inverse_zero(self, G: MatrixLieGroup):
        xi = np.zeros((G.dof, 1))
        J_left = G.left_jacobian(xi)
        J_left_inv = G.left_jacobian_inv(xi)
        assert not np.isnan(J_left_inv).any()
        assert np.allclose(J_left_inv, np.linalg.inv(J_left))

    def test_left_jacobian_inverse_small_value(self, G: MatrixLieGroup):
        xi = np.zeros((G.dof, 1))
        xi[0] = 1e-8
        J_left = G.left_jacobian(xi)
        J_left_inv = G.left_jacobian_inv(xi)
        assert not np.isnan(J_left_inv).any()
        assert np.allclose(J_left_inv, np.linalg.inv(J_left))

    def test_right_jacobian_inverse(self, G: MatrixLieGroup):
        X = G.random()
        xi = G.Log(X)
        J_right = G.right_jacobian(xi)
        J_right_inv = G.right_jacobian_inv(xi)

        assert np.allclose(J_right_inv, np.linalg.inv(J_right))

    def test_left_jacobian(self, G: MatrixLieGroup):
        np.random.seed(0)
        x_bar = G.Log(G.random())
        J_left = G.left_jacobian(x_bar)
        J_fd = self._numerical_left_jacobian(G, x_bar)

        assert np.allclose(J_fd, J_left, atol=1e-2)

    def test_left_jacobian_zero(self, G: MatrixLieGroup):
        x_bar = np.zeros((G.dof, 1))
        J_left = G.left_jacobian(x_bar)
        J_fd = self._numerical_left_jacobian(G, x_bar)

        assert np.allclose(J_fd, J_left, atol=1e-5)

    
    def test_left_jacobian_small_value(self, G: MatrixLieGroup):
        x_bar = np.zeros((G.dof, 1))
        x_bar[0] = 1e-8
        J_left = G.left_jacobian(x_bar)
        J_fd = self._numerical_left_jacobian(G, x_bar)

        assert np.allclose(J_fd, J_left, atol=1e-5)

    def _numerical_left_jacobian(self, G: MatrixLieGroup, x_bar: np.ndarray):
        exp_inv = G.inverse(G.Exp(x_bar))
        J_fd = np.zeros((G.dof, G.dof))
        h = 1e-7
        for i in range(G.dof):
            dx = np.zeros((G.dof, 1))
            dx[i] = h
            J_fd[:, i] = (G.Log(np.dot(G.Exp(x_bar + dx), exp_inv)) / h).ravel()

        return J_fd


    def test_adjoint_identity(self, G: MatrixLieGroup):
        X = G.random()
        xi = G.Log(G.random())

        side1 = G.wedge(np.dot(G.adjoint(X), xi))
        side2 = np.dot(X, np.dot(G.wedge(xi), G.inverse(X)))
        assert np.allclose(side1, side2)

    def test_adjoint_algebra_identity(self, G: MatrixLieGroup):
        Xi1 = G.log(G.random())
        Xi2 = G.log(G.random())
        xi1 = np.atleast_2d(G.vee(Xi1))
        xi2 = np.atleast_2d(G.vee(Xi2))

        assert np.allclose(G.adjoint_algebra(Xi1) @ xi2, -G.adjoint_algebra(Xi2) @ xi1)

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
        self.test_left_jacobian(G) 
        self.test_left_jacobian_small_value(G)
        self.test_left_jacobian_inverse(G)
        self.test_right_jacobian_inverse(G)
        self.test_left_jacobian(G)
        self.test_adjoint_identity(G)
        self.test_adjoint_algebra_identity(G)
        self.test_inverse(G)

class CrossValidation:
    def test_wedge(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        x = np.random.random((G1.dof, 1))
        assert np.allclose(G1.wedge(x), G2.wedge(x))

    def test_exp(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        x = np.random.random((G1.dof, 1))
        Xi = G1.wedge(x)
        assert np.allclose(G1.exp(Xi), G2.exp(Xi))

    def test_log(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        X = G1.random()
        assert np.allclose(G1.log(X), G2.log(X))

    def test_capital_exp(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        x = np.random.random((G1.dof, 1))
        assert np.allclose(G1.Exp(x), G2.Exp(x))
    
    def test_capital_log(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        X = G1.random()
        assert np.allclose(G1.Log(X), G2.Log(X))

    def test_odot(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        b = np.random.random((G1.matrix_size, 1))
        assert np.allclose(G1.odot(b), G2.odot(b))

    def test_left_jacobian(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        x = np.random.random((G1.dof, 1))
        assert np.allclose(G1.left_jacobian(x), G2.left_jacobian(x), atol=1e-7)
    
    def test_right_jacobian(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        x = np.random.random((G1.dof, 1))
        assert np.allclose(G1.right_jacobian(x), G2.right_jacobian(x), atol=1e-7)

    def test_left_jacobian_inv(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        x = np.random.random((G1.dof, 1))
        assert np.allclose(G1.left_jacobian_inv(x), G2.left_jacobian_inv(x), atol=1e-7)
    
    def test_right_jacobian_inv(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        x = np.random.random((G1.dof, 1))
        assert np.allclose(G1.right_jacobian_inv(x), G2.right_jacobian_inv(x), atol=1e-7)
    
    def test_adjoint(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        X = G1.random()
        assert np.allclose(G1.adjoint(X), G2.adjoint(X))
    
    def test_adjoint_algebra(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        X = G1.random()
        Xi = G1.log(X)
        assert np.allclose(G1.adjoint_algebra(Xi), G2.adjoint_algebra(Xi))

    def test_inverse(self, G1: MatrixLieGroup, G2: MatrixLieGroup):
        X = G1.random()
        assert np.allclose(G1.inverse(X), G2.inverse(X))
