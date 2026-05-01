import numpy as np
import torch
from random import randrange
from scipy.linalg import expm, logm
from pymlg.torch import MatrixLieGroupTorch

# set pytorch to double precision for testing
torch.set_default_dtype(torch.float64)

# TODO: np.allclose requires a CPU tensor, so to avoimd moving it to the CPU every time, implement something that checks whether the target device is CPU or GPU and uses the appropriate allclose method

class StandardTestsTorch:
    def test_wedge_vee(self, G: MatrixLieGroupTorch, device):
        x = torch.rand(randrange(1, 10), G.dof, 1)
        x = x.to(device)
        x_test = G.vee(G.wedge(x))
        if G.dof > 1:
            assert x_test.shape == (x.shape[0], G.dof, 1)

        # change allclose assertion based on whether the inputs are torch tensors or numpy arrays
        if x.dtype == torch.float32 or x.dtype == torch.float64:
            assert torch.allclose(x, x_test, 1e-15)
        else:
            assert np.allclose(x, x_test, 1e-15)

    def test_exp(self, G: MatrixLieGroupTorch, device):
        x = torch.rand(randrange(1, 10), G.dof, 1)
        x = x.to(device)
        Xi = G.wedge(x)
        X = G.exp(Xi)
        if Xi.dtype == torch.float32 or Xi.dtype == torch.float64:
            Xi = Xi.cpu().numpy()
            X = X.cpu().numpy()
        Xi = np.array(Xi).copy()
        X_test = expm(Xi)
        assert np.allclose(X, X_test)

    def test_log(self, G: MatrixLieGroupTorch, device):
        X = G.random(device=device)
        Xi = G.log(X)
        if X.dtype == torch.float32 or X.dtype == torch.float64:
            X = X.cpu().numpy()
            Xi = Xi.cpu().numpy()
        X = np.array(X).copy()
        
        # note. logm is not currently batched as per (https://github.com/scipy/scipy/issues/12838#issuecomment-1539746877), but the G.random() call for all classes is single-batch. So by definition, should be safe to squeeze the batch dimension and test with unbatched logm().
        Xi_test = logm(X.squeeze(0))
        assert np.allclose(Xi, Xi_test)

    def test_log_zero(self, G: MatrixLieGroupTorch, device):
        x = torch.zeros(1, G.dof, 1)
        x = x.to(device)
        X = G.Exp(x)
        Xi = G.log(X)
        if X.dtype == torch.float32 or X.dtype == torch.float64:
            X = X.cpu().numpy()
            Xi = Xi.cpu().numpy()
        X = np.array(X).copy()
        Xi_test = logm(X.squeeze(0))
        assert np.allclose(Xi, Xi_test)

    def test_capital_log_zero(self, G: MatrixLieGroupTorch, device):
        x = torch.zeros(randrange(1, 10), G.dof, 1)
        x = x.to(device)
        X = G.Exp(x)
        x_test = G.Log(X)
        assert torch.allclose(x, x_test)

    def test_capital_log_small_value(self, G: MatrixLieGroupTorch, device):
        x = torch.zeros(randrange(1, 10), G.dof, 1)
        x = x.to(device)
        x[0] = 1e-8
        X = G.Exp(x)
        x_test = G.Log(X)
        assert not torch.isnan(x_test).any()
        assert torch.allclose(x, x_test)

    def test_exp_log_inverse(self, G: MatrixLieGroupTorch, device):
        X = G.random(device=device)
        Xi = G.log(X)
        assert torch.allclose(X, G.exp(G.log(X)))
        assert torch.allclose(Xi, G.log(G.exp(Xi)))

    def test_capital_exp_log_inverse(self, G: MatrixLieGroupTorch, device):
        T = G.random(device=device)
        x = G.Log(T)
        assert torch.allclose(T, G.Exp(x))

        if G.dof > 1:
            assert x.shape == (1, G.dof, 1)

    def test_odot_wedge(self, G: MatrixLieGroupTorch, device):
        X = G.random(device=device)
        a = G.Log(X)
        b = torch.normal(0, 1, (X.shape[0], X.shape[1], 1), device=device)

        test1 = G.wedge(a) @ b
        test2 = G.odot(b) @ a
        assert torch.allclose(test1, test2)

    def test_left_jacobian_inverse(self, G: MatrixLieGroupTorch, device):
        X = G.random(device=device)
        xi = G.Log(X)
        J_left = G.left_jacobian(xi)
        J_left_inv = G.left_jacobian_inv(xi)

        assert torch.allclose(J_left_inv, torch.linalg.inv(J_left))

    
    def test_left_jacobian_inverse_zero(self, G: MatrixLieGroupTorch, device):
        xi = torch.zeros(randrange(1, 10), G.dof, 1)
        xi = xi.to(device)
        J_left = G.left_jacobian(xi)
        J_left_inv = G.left_jacobian_inv(xi)
        assert not torch.isnan(J_left_inv).any()
        assert torch.allclose(J_left_inv, torch.linalg.inv(J_left))

    def test_left_jacobian_inverse_small_value(self, G: MatrixLieGroupTorch, device):
        xi = torch.zeros(randrange(1, 10), G.dof, 1)
        xi = xi.to(device)
        xi[0] = 1e-8
        J_left = G.left_jacobian(xi)
        J_left_inv = G.left_jacobian_inv(xi)
        assert not torch.isnan(J_left_inv).any()
        assert torch.allclose(J_left_inv, torch.linalg.inv(J_left))

    def test_right_jacobian_inverse(self, G: MatrixLieGroupTorch, device):
        X = G.random(device=device)
        xi = G.Log(X)
        J_right = G.right_jacobian(xi)
        J_right_inv = G.right_jacobian_inv(xi)

        assert torch.allclose(J_right_inv, torch.linalg.inv(J_right))

    def test_left_jacobian(self, G: MatrixLieGroupTorch, device):
        x_bar = G.Log(G.random(device=device))
        J_left = G.left_jacobian(x_bar)
        J_fd = self._numerical_left_jacobian(G, x_bar)

        assert torch.allclose(J_fd, J_left, atol=1e-2)

    def test_left_jacobian_zero(self, G: MatrixLieGroupTorch, device):
        x_bar = torch.zeros(randrange(1, 10), G.dof, 1)
        x_bar = x_bar.to(device)
        J_left = G.left_jacobian(x_bar)
        J_fd = self._numerical_left_jacobian(G, x_bar)

        assert torch.allclose(J_fd, J_left, atol=1e-5)

    
    def test_left_jacobian_small_value(self, G: MatrixLieGroupTorch, device):
        x_bar = torch.zeros(randrange(1, 10), G.dof, 1)
        x_bar = x_bar.to(device)
        x_bar[0] = 1e-8
        J_left = G.left_jacobian(x_bar)
        J_fd = self._numerical_left_jacobian(G, x_bar)

        assert torch.allclose(J_fd, J_left, atol=1e-5)

    def _numerical_left_jacobian(self, G: MatrixLieGroupTorch, x_bar: torch.Tensor):
        device = x_bar.device
        exp_inv = G.inverse(G.Exp(x_bar))
        J_fd = torch.zeros(x_bar.shape[0], G.dof, G.dof, device=device) #np.zeros((G.dof, G.dof))
        h = 1e-7
        for i in range(G.dof):
            dx = torch.zeros(x_bar.shape[0], G.dof, 1, device=device) #np.zeros((G.dof, 1))
            dx[:, i, :] = h
            J_fd[:, :, i] = (G.Log(G.Exp(x_bar + dx) @ exp_inv) / h).squeeze(2)

        return J_fd

    def test_adjoint_identity(self, G: MatrixLieGroupTorch, device):
        X = G.random(device=device)
        xi = G.Log(G.random(device=device))

        side1 = G.wedge(G.adjoint(X) @ xi)
        side2 = X @ (G.wedge(xi) @ G.inverse(X)) #np.dot(X, np.dot(G.wedge(xi), G.inverse(X)))
        assert torch.allclose(side1, side2)

    def test_adjoint_algebra_identity(self, G: MatrixLieGroupTorch, device):
        Xi1 = G.log(G.random(device=device))
        Xi2 = G.log(G.random(device=device))
        xi1 = G.vee(Xi1)
        xi2 = G.vee(Xi2)

        assert torch.allclose(G.adjoint_algebra(Xi1) @ xi2, -G.adjoint_algebra(Xi2) @ xi1)

    def test_inverse(self, G: MatrixLieGroupTorch, device):
        X = G.random(device=device)
        assert torch.allclose(G.inverse(G.inverse(X)), X)
        assert torch.allclose(G.inverse(X), torch.linalg.inv(X))
        assert torch.allclose(G.inverse(G.identity(device=device)), G.identity(device=device))
        assert torch.allclose(G.inverse(X) @ X, G.identity(device=device))

    def do_tests(self, G: MatrixLieGroupTorch, test_device):
        self.test_wedge_vee(G, device=test_device)
        self.test_exp(G, device=test_device)
        self.test_log(G, device=test_device)
        self.test_exp_log_inverse(G, device=test_device)
        self.test_capital_exp_log_inverse(G, device=test_device)
        self.test_odot_wedge(G, device=test_device)
        self.test_left_jacobian(G, device=test_device) 
        self.test_left_jacobian_small_value(G, device=test_device)
        self.test_left_jacobian_inverse(G, device=test_device)
        self.test_right_jacobian_inverse(G, device=test_device)
        self.test_left_jacobian(G, device=test_device)
        self.test_adjoint_identity(G, device=test_device)
        self.test_adjoint_algebra_identity(G, device=test_device)
        self.test_inverse(G, device=test_device)

class CrossValidation:
    def test_wedge(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.arange(0, G1.dof).unsqueeze(0) * .1
        assert torch.allclose(G1.wedge(x), G2.wedge(x))

    def test_exp(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        Xi = G1.wedge(x)
        assert torch.allclose(G1.exp(Xi), G2.exp(Xi))

    def test_log(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        assert torch.allclose(G1.log(X), G2.log(X))

    def test_capital_exp(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert torch.allclose(G1.Exp(x), G2.Exp(x))
    
    def test_capital_log(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        assert torch.allclose(G1.Log(X), G2.Log(X))

    def test_odot(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        b = torch.rand(1, G1.matrix_size, 1)
        assert torch.allclose(G1.odot(b), G2.odot(b))

    def test_left_jacobian(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert torch.allclose(G1.left_jacobian(x), G2.left_jacobian(x), atol=1e-6)
    
    def test_right_jacobian(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert torch.allclose(G1.right_jacobian(x), G2.right_jacobian(x), atol=1e-6)

    def test_left_jacobian_inv(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert torch.allclose(G1.left_jacobian_inv(x), G2.left_jacobian_inv(x), atol=1e-6)
    
    def test_right_jacobian_inv(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        x = torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1)
        assert torch.allclose(G1.right_jacobian_inv(x), G2.right_jacobian_inv(x), atol=1e-6)
    
    def test_adjoint(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        assert torch.allclose(G1.adjoint(X), G2.adjoint(X))
    
    def test_adjoint_algebra(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        Xi = G1.log(X)
        assert torch.allclose(G1.adjoint_algebra(Xi), G2.adjoint_algebra(Xi))

    def test_inverse(self, G1: MatrixLieGroupTorch, G2: MatrixLieGroupTorch):
        X = G1.Exp(torch.linspace(.1, 1, G1.dof).view(1, G1.dof, 1))
        assert torch.allclose(G1.inverse(X), G2.inverse(X))
