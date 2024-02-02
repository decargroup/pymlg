from .base import MatrixLieGroup
import torch
from .utils import *
from .so2 import SO2

class SE2(MatrixLieGroup):
    """
    Special Euclidean Group in 2D.
    """

    dof = 3
    matrix_size = 3

    @staticmethod
    def random():
        phi = torch.rand(1, 1) * 2 * torch.pi
        r = torch.randn(2, 1)
        C = SO2.Exp(phi)
        return SE2.from_components(C, r)    
    
    @staticmethod
    def from_components(C, r):
        """
        Construct an SE(2) matrix from a rotation matrix and translation vector.
        """
        T = torch.zeros(C.shape[0], 3, 3)

        T[:, 0:2, 0:2] = C
        T[:, 0:2, 2] = r.view(-1, 2)
        T[:, 2, 2] = 1

        return T
    
    @staticmethod
    def to_components(T):
        """
        Decompose an SE(2) matrix into a rotation matrix and translation vector.
        """
        C = T[:, 0:2, 0:2]
        r = T[:, 0:2, 2]
        return C, r

    @staticmethod
    def wedge(xi):
        phi = xi[:, 0]
        xi_r = xi[:, 1:]
        Xi_phi = SO2.wedge(phi)
        Xi = torch.zeros(xi.shape[0], 3, 3)
        Xi[:, 0:2, 0:2] = Xi_phi
        Xi[:, 0:2, 2] = xi_r.view(-1, 2)
        return Xi
    
    @staticmethod
    def vee(Xi):
        Xi_phi = Xi[:, 0:2, 0:2]
        xi_r = Xi[:, 0:2, 2].unsqueeze(2)
        phi = SO2.vee(Xi_phi)
        xi = torch.cat((phi, xi_r), dim=1)
        return xi
    
    @staticmethod
    def exp(Xi):
        Xi_phi = Xi[:, 0:2, 0:2]
        xi_r = Xi[:, 0:2, 2].unsqueeze(2)
        phi = SO2.vee(Xi_phi)
        C = SO2.Exp(phi)
        r = SE2.V_matrix(phi.view(-1, 1)) @ xi_r
        return SE2.from_components(C, r)
    
    @staticmethod
    def log(T):
        Xi_phi = SO2.log(T[:, 0:2, 0:2])
        r = T[:, 0:2, 2].unsqueeze(2)
        xi_r = SE2.V_matrix_inv(SO2.vee(Xi_phi)) @ r
        Xi = torch.zeros(T.shape[0], 3, 3)
        Xi[:, 0:2, 0:2] = Xi_phi
        Xi[:, 0:2, 2] = xi_r.squeeze(2)
        return Xi
    
    @staticmethod
    def odot(b):

        X = torch.zeros(b.shape[0], 3, 3)
        X[:, 0:2, 0] = SO2.odot(b[:, :2]).squeeze(2)
        X[:, 0:2, 1:3] = batch_eye(b.shape[0], 2, 2) * b[:, 2].unsqueeze(2)

        return X
    
    @staticmethod
    def left_jacobian(xi):

        # enforce dimensionality
        
        rho = xi[:, 1:]
        phi = xi[:, 0]

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        phi_sq = phi ** 2

        small_angle_mask = is_close(phi_sq, 1e-15)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=True)[0]
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=True)[0]

        J = torch.zeros(xi.shape[0], 3, 3)

        if small_angle_inds.numel():
            A = (1 - 1.0 / 6.0 * phi_sq[small_angle_inds]).view(-1)
            B = (0.5 * phi[small_angle_inds] - 1.0 / 24.0 * phi[small_angle_inds] * phi_sq[small_angle_inds]).view(-1)
            J[small_angle_inds, 1, 1] = A
            J[small_angle_inds, 2, 2] = A
            J[small_angle_inds, 1, 2] = -B
            J[small_angle_inds, 2, 1] = B

            C = (rho[small_angle_inds, 1] / 2.0 + phi[small_angle_inds] * rho[small_angle_inds, 0] / 6.0).view(-1)
            D = (-rho[small_angle_inds, 0] / 2.0 + phi[small_angle_inds] * rho[small_angle_inds, 1] / 6.0).view(-1)

            J[small_angle_inds, 1, 0] = C
            J[small_angle_inds, 2, 0] = D

        if large_angle_inds.numel():
            A = sin_phi[large_angle_inds] / phi[large_angle_inds]
            B = (1 - cos_phi[large_angle_inds]) / phi[large_angle_inds]
            J[large_angle_inds, 1, 1] = A
            J[large_angle_inds, 2, 2] = A
            J[large_angle_inds, 1, 2] = -B
            J[large_angle_inds, 2, 1] = B

            C = (rho[large_angle_inds, 1] + phi[large_angle_inds] * rho[large_angle_inds, 0] - rho[large_angle_inds, 1] * cos_phi[large_angle_inds] - rho[large_angle_inds, 0] * sin_phi[large_angle_inds]) / (phi_sq[large_angle_inds])

            D = (-rho[large_angle_inds, 0] + phi[large_angle_inds] * rho[large_angle_inds, 1] + rho[large_angle_inds, 0] * cos_phi[large_angle_inds] - rho[large_angle_inds, 1] * sin_phi[large_angle_inds]) / (phi_sq[large_angle_inds])

            J[large_angle_inds, 1, 0] = C
            J[large_angle_inds, 2, 0] = D

        J[:, 0, 0] = 1

        return J
    
    @staticmethod
    def adjoint(T):
        C = T[:, 0:2, 0:2]
        r = T[:, 0:2, 2]

        # build Om matrix manually (will this break the DAG?)
        Om = torch.Tensor([[0, -1], [1, 0]]).repeat(T.shape[0], 1, 1)

        A = torch.zeros(T.shape[0], 3, 3)
        A[:, 0, 0] = 1
        A[:, 1:, 0] = -(Om @ r.unsqueeze(2)).squeeze(2)
        A[:, 1:, 1:] = C

        return A
    
    @staticmethod
    def adjoint_algebra(Xi):
        A = torch.zeros(Xi.shape[0], 3, 3)
        A[:, 1, 0] = Xi[:, 1, 2]
        A[:, 2, 0] = -Xi[:, 0, 2]
        A[:, 1:, 1:] = Xi[:, 0:2, 0:2]
        return A
    
    @staticmethod
    def V_matrix(phi):

        phi_norm = torch.linalg.norm(phi, dim=1)

        small_angle_mask = is_close(phi_norm, 0.0)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=True)[0]
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=True)[0]

        V = batch_eye(phi.shape[0], 2, 2)

        if small_angle_inds.numel():
            V[small_angle_inds] += .5 * SO2.wedge(phi[small_angle_inds])
            
        if large_angle_inds.numel():
            s = torch.sin(phi[large_angle_inds])
            c = torch.cos(phi[large_angle_inds])

            V[large_angle_inds]  = V[large_angle_inds] * (s / phi[large_angle_inds]).view(-1, 1, 1) + ((1 - c) / phi[large_angle_inds]).view(-1, 1, 1) * SO2.wedge(torch.ones(phi[large_angle_inds].shape[0]))

        return V
    
    @staticmethod
    def V_matrix_inv(phi):
        phi_norm = torch.linalg.norm(phi, dim=1)

        small_angle_mask = is_close(phi_norm, 0.0)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=True)[0]
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=True)[0]

        V_inv = batch_eye(phi.shape[0], 2, 2)

        if small_angle_inds.numel():
            V_inv[small_angle_inds] -= .5 * SO2.wedge(phi[small_angle_inds])

        if large_angle_inds.numel():
            half_angle = phi[large_angle_inds] / 2
            cot_half_angle = 1 / torch.tan(half_angle)
            V_inv[large_angle_inds] = V_inv[large_angle_inds] * half_angle * cot_half_angle - half_angle * SO2.wedge(torch.ones(half_angle.shape[0]))

        return V_inv



