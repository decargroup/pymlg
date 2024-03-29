import numpy as np
import torch
from random import randrange
from scipy.linalg import expm, logm
from pymlg.torch import MatrixLieGroupTorch

# set pytorch to double precision for testing
torch.set_default_dtype(torch.float64)

class StandardTestsTorch:
    def test_wedge_vee(self, G: MatrixLieGroupTorch):
        x = torch.rand(randrange(1, 10), G.dof, 1)
        x_test = G.vee(G.wedge(x))
        if G.dof > 1:
            assert x_test.shape == (x.shape[0], G.dof, 1)
        assert np.allclose(x, x_test, 1e-15)

    def test_exp(self, G: MatrixLieGroupTorch):
        x = torch.rand(randrange(1, 10), G.dof, 1)
        Xi = G.wedge(x)
        X = G.exp(Xi)
        Xi = np.array(Xi).copy()
        X_test = expm(Xi)
        assert np.allclose(X, X_test)

    def test_log(self, G: MatrixLieGroupTorch):
        X = G.random()
        Xi = G.log(X)
        X = np.array(X).copy()
        
        # note. logm is not currently batched as per (https://github.com/scipy/scipy/issues/12838#issuecomment-1539746877), but the G.random() call for all classes is single-batch. So by definition, should be safe to squeeze the batch dimension and test with unbatched logm().
        Xi_test = logm(X.squeeze(0))
        assert np.allclose(Xi, Xi_test)

    def test_log_zero(self, G: MatrixLieGroupTorch):
        x = torch.zeros(1, G.dof, 1)
        X = G.Exp(x)
        Xi = G.log(X)
        X = np.array(X).copy()
        Xi_test = logm(X.squeeze(0))
        assert np.allclose(Xi, Xi_test)

    def test_capital_log_zero(self, G: MatrixLieGroupTorch):
        x = torch.zeros(randrange(1, 10), G.dof, 1)
        X = G.Exp(x)
        x_test = G.Log(X)
        assert np.allclose(x, x_test)

    def test_capital_log_small_value(self, G: MatrixLieGroupTorch):
        x = torch.zeros(randrange(1, 10), G.dof, 1)
        x[0] = 1e-8
        X = G.Exp(x)
        x_test = G.Log(X)
        assert not np.isnan(x_test).any()
        assert np.allclose(x, x_test)

    def test_exp_log_inverse(self, G: MatrixLieGroupTorch):
        X = G.random()
        Xi = G.log(X)
        assert np.allclose(X, G.exp(G.log(X)))
        assert np.allclose(Xi, G.log(G.exp(Xi)))

    def test_capital_exp_log_inverse(self, G: MatrixLieGroupTorch):
        T = G.random()
        x = G.Log(T)
        assert np.allclose(T, G.Exp(x))

        if G.dof > 1:
            assert x.shape == (1, G.dof, 1)

    def test_odot_wedge(self, G: MatrixLieGroupTorch):
        X = G.random()
        a = G.Log(X)
        b = torch.normal(0, 1, (X.shape[0], X.shape[1], 1))

        test1 = G.wedge(a) @ b
        test2 = G.odot(b) @ a
        assert np.allclose(test1, test2)

    def test_left_jacobian_inverse(self, G: MatrixLieGroupTorch):
        X = G.random()
        xi = G.Log(X)
        J_left = G.left_jacobian(xi)
        J_left_inv = G.left_jacobian_inv(xi)

        assert np.allclose(J_left_inv, torch.linalg.inv(J_left))

    
    def test_left_jacobian_inverse_zero(self, G: MatrixLieGroupTorch):
        xi = torch.zeros(randrange(1, 10), G.dof, 1)
        J_left = G.left_jacobian(xi)
        J_left_inv = G.left_jacobian_inv(xi)
        assert not np.isnan(J_left_inv).any()
        assert np.allclose(J_left_inv, np.linalg.inv(J_left))

    def test_left_jacobian_inverse_small_value(self, G: MatrixLieGroupTorch):
        xi = torch.zeros(randrange(1, 10), G.dof, 1)
        xi[0] = 1e-8
        J_left = G.left_jacobian(xi)
        J_left_inv = G.left_jacobian_inv(xi)
        assert not np.isnan(J_left_inv).any()
        assert np.allclose(J_left_inv, np.linalg.inv(J_left))

    def test_right_jacobian_inverse(self, G: MatrixLieGroupTorch):
        X = G.random()
        xi = G.Log(X)
        J_right = G.right_jacobian(xi)
        J_right_inv = G.right_jacobian_inv(xi)

        assert np.allclose(J_right_inv, np.linalg.inv(J_right))

    def test_left_jacobian(self, G: MatrixLieGroupTorch):
        x_bar = G.Log(G.random())
        J_left = G.left_jacobian(x_bar)
        J_fd = self._numerical_left_jacobian(G, x_bar)

        assert np.allclose(J_fd, J_left, atol=1e-2)

    def test_left_jacobian_zero(self, G: MatrixLieGroupTorch):
        x_bar = torch.zeros(randrange(1, 10), G.dof, 1)
        J_left = G.left_jacobian(x_bar)
        J_fd = self._numerical_left_jacobian(G, x_bar)

        assert np.allclose(J_fd, J_left, atol=1e-5)

    
    def test_left_jacobian_small_value(self, G: MatrixLieGroupTorch):
        x_bar = torch.zeros(randrange(1, 10), G.dof, 1)
        x_bar[0] = 1e-8
        J_left = G.left_jacobian(x_bar)
        J_fd = self._numerical_left_jacobian(G, x_bar)

        assert np.allclose(J_fd, J_left, atol=1e-5)

    def _numerical_left_jacobian(self, G: MatrixLieGroupTorch, x_bar: torch.Tensor):
        exp_inv = G.inverse(G.Exp(x_bar))
        J_fd = torch.zeros(x_bar.shape[0], G.dof, G.dof) #np.zeros((G.dof, G.dof))
        h = 1e-7
        for i in range(G.dof):
            dx = torch.zeros(x_bar.shape[0], G.dof, 1)
            dx[:, i, :] = h
            J_fd[:, :, i] = (G.Log(G.Exp(x_bar + dx) @ exp_inv) / h).squeeze(2)

        return J_fd

    def test_adjoint_identity(self, G: MatrixLieGroupTorch):
        X = G.random()
        xi = G.Log(G.random())

        side1 = G.wedge(G.adjoint(X) @ xi)
        side2 = X @ (G.wedge(xi) @ G.inverse(X)) #np.dot(X, np.dot(G.wedge(xi), G.inverse(X)))
        assert np.allclose(side1, side2)

    def test_adjoint_algebra_identity(self, G: MatrixLieGroupTorch):
        Xi1 = G.log(G.random())
        Xi2 = G.log(G.random())
        xi1 = G.vee(Xi1)
        xi2 = G.vee(Xi2)

        assert np.allclose(G.adjoint_algebra(Xi1) @ xi2, -G.adjoint_algebra(Xi2) @ xi1)

    def test_inverse(self, G: MatrixLieGroupTorch):
        X = G.random()
        assert np.allclose(G.inverse(G.inverse(X)), X)
        assert np.allclose(G.inverse(X), np.linalg.inv(X))
        assert np.allclose(G.inverse(G.identity()), G.identity())
        assert np.allclose(G.inverse(X) @ X, G.identity())

    def do_tests(self, G: MatrixLieGroupTorch):
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
    def test_wedge(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.arange(0, G1.dof).unsqueeze(0) * .1
        assert np.allclose(G1.wedge(x), G2.wedge(x))

    def test_exp(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        Xi = G1.wedge(x)
        assert np.allclose(G1.exp(Xi), G2.exp(Xi))

    def test_log(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        assert np.allclose(G1.log(X), G2.log(X))

    def test_capital_exp(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert np.allclose(G1.Exp(x), G2.Exp(x))
    
    def test_capital_log(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        assert np.allclose(G1.Log(X), G2.Log(X))

    def test_odot(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        b = torch.rand(1, G1.matrix_size, 1)
        assert np.allclose(G1.odot(b), G2.odot(b))

    def test_left_jacobian(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert np.allclose(G1.left_jacobian(x), G2.left_jacobian(x), atol=1e-6)
    
    def test_right_jacobian(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert np.allclose(G1.right_jacobian(x), G2.right_jacobian(x), atol=1e-6)

    def test_left_jacobian_inv(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert np.allclose(G1.left_jacobian_inv(x), G2.left_jacobian_inv(x), atol=1e-6)
    
    def test_right_jacobian_inv(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert np.allclose(G1.right_jacobian_inv(x), G2.right_jacobian_inv(x), atol=1e-6)
    
    def test_adjoint(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        assert np.allclose(G1.adjoint(X), G2.adjoint(X))
    
    def test_adjoint_algebra(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        Xi = G1.log(X)
        assert np.allclose(G1.adjoint_algebra(Xi), G2.adjoint_algebra(Xi))

    def test_inverse(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        assert np.allclose(G1.inverse(X), G2.inverse(X))
