from .base import MatrixLieGroup
import torch
from .utils import *

class SO2(MatrixLieGroup):
    """
    Special Orthogonal Group in 2D
    """

    dof = 1
    matrix_size = 2

    @staticmethod
    def random(N=1):
        """
        Generates a random batch of SO_(2) matricies.

        Parameters
        ----------
        N : int, optional
            batch size, by default 1
        """
        phi = torch.rand(N, 1) * 2 * torch.pi
        return SO2.Exp(phi)
    
    @staticmethod
    def wedge(phi):

        X = torch.zeros(phi.shape[0], 2, 2, device=phi.device)
        X[:, 0, 1] = -phi
        X[:, 1, 0] = phi

        return X
    
    @staticmethod
    def vee(X):
        phi = X[:, 1, 0]
        return phi
    
    @staticmethod
    def exp(Xi):
        phi = SO2.vee(Xi)
        X = torch.zeros(Xi.shape[0], 2, 2, device=Xi.device)
        X[:, 0, 0] = torch.cos(phi)
        X[:, 0, 1] = -torch.sin(phi)
        X[:, 1, 0] = torch.sin(phi)
        X[:, 1, 1] = torch.cos(phi)
        return X
    
    @staticmethod
    def log(X):
        phi = torch.atan2(X[:, 1, 0], X[:, 0, 0])
        return SO2.wedge(phi)
    
    @staticmethod
    def left_jacobian(x):
        return torch.ones(x.shape[0], 1, 1, device=x.device)
    
    @staticmethod
    def left_jacobian_inv(x):
        return torch.ones(x.shape[0], 1, 1, device=x.device)
    
    @staticmethod
    def right_jacobian(x):
        return torch.ones(x.shape[0], 1, 1, device=x.device)
    
    @staticmethod
    def right_jacobian_inv(x):
        return torch.ones(x.shape[0], 1, 1, device=x.device)
    
    @staticmethod
    def adjoint(C):
        return torch.ones(C.shape[0], 1, 1, device=C.device)
    
    @staticmethod
    def adjoint_algebra(Xi):
        return torch.zeros(Xi.shape[0], 1, 1, device=Xi.device)
    
    @staticmethod
    def odot(xi):
        X = torch.zeros(xi.shape[0], 2, 1, device=xi.device)
        X[:, 0] = xi[:, 1]
        X[:, 1] = xi[:, 0]
        return X