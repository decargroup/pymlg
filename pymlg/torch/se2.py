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
        T[:, 0:2, 2] = r

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
        Xi[:, 0:2, 2] = xi_r
        return Xi
    
    @staticmethod
    def vee(Xi):
        Xi_phi = Xi[:, 0:2, 0:2]
        xi_r = Xi[:, 0:2, 2]
        phi = SO2.vee(Xi_phi)
        xi = torch.cat((phi, xi_r), dim=1)
        return xi
    
    @staticmethod
    def exp(Xi):
        Xi_phi = Xi[:, 0:2, 0:2]
        xi_r = Xi[:, 0:2, 2]
        phi = SO2.vee(Xi_phi)
        C = SO2.Exp(phi)
        r = SE2.V_matrix(phi) @ xi_r
        return SE2.from_components(C, r)
    
    @staticmethod
    def log(T):
        Xi_phi = SO2.log(T[:, 0:2, 0:2])
        r = T[:, 0:2, 2]
        xi_r = SE2.V_matrix_inv(SO2.vee(Xi_phi)) @ r
        Xi = torch.zeros(T.shape[0], 3, 3)
        Xi[:, 0:2, 0:2] = Xi_phi
        Xi[:, 0:2, 2] = xi_r
        return Xi
    
    @staticmethod
    def odot(b):

        X = torch.zeros(b.shape[0], 3, 3)
        X[:, 0:2, 0] = SO2.odot(b[:, :2])
        X[:, 0:2, 1:3] = b[:, 2] @ batch_eye(b.shape[0], 2, 2)

        return X
    
    @staticmethod
    def left_jacobian(xi):
        raise NotImplementedError("Left jacobian not implemented for SE2!")
    
    @staticmethod
    def left_jacobian_inv(xi):
        raise NotImplementedError("Left jacobian inverse not implemented for SE2!")
    
    @staticmethod
    def adjoint(T):
        C = T[:, 0:2, 0:2]
        r = T[:, 0:2, 2]

        # build Om matrix manually (will this break the DAG?)
        Om = torch.Tensor([[0, 1], [-1, 0]]).repeat(T.shape[0], 1, 1)

        A = torch.zeros(T.shape[0], 3, 3)
        A[:, 0, 0] = 1
        A[:, 1:, 0] = -Om @ r
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

            V[large_angle_inds]  = V[large_angle_inds] * (s / phi[large_angle_inds]) + ((1 - c) / phi[large_angle_inds]) @ SO2.wedge(torch.ones(phi[large_angle_inds].shape[0]))

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



