from .base import MatrixLieGroup
import torch
from .utils import *
from .so3 import SO3
from .se3 import SE3

# DISCLAIMER: this class is very much so un-tested


class SE23(MatrixLieGroup):
    """
    An torch instantiation-free implementation of the SE_2(3) matrix Lie group.

    ALL functions are made to accept a batch: [N x ...].
    """

    dof = 9

    @staticmethod
    def from_components(C: torch.Tensor, v: torch.Tensor, r: torch.Tensor):
        """
        Construct a batch of :math:`SE_2(3)` matricies from attitude, velocity, position
        components.

        Parameters
        ----------
        C : torch.Tensor with shape (N, 3, 3)
            batch of rotation matricies/DCMs
        v : torch.Tensor with size (N, 3)
            batch of velocity vectors
        r : torch.Tensor with size (N, 3)
            batch of position vectors

        Returns
        -------
        torch.Tensor with shape (N, 5, 5)
            :math:`SE_2(3)` matrix
        """

        # firstly, check that batch dimension for all 3 components matches
        if not (C.shape[0] == v.shape[0] == r.shape[0]):
            raise ValueError("Batch dimension for SE_2(3) components don't match.")

        X = batch_eye(C.shape[0], 5, 5)

        X[:, 0:3, 0:3] = C
        X[:, 0:3, 3] = v.squeeze(2)
        X[:, 0:3, 4] = r.squeeze(2)

        return X

    @staticmethod
    def to_components(X: torch.Tensor):
        """
        extract rotation, velocity, and position components from SE_2(3) batch
        """
        C = X[:, 0:3, 0:3]
        v = X[:, 0:3, 3]
        r = X[:, 0:3, 4]

        # enforce dimensionality
        return (C, v.unsqueeze(2), r.unsqueeze(2))
    
    @staticmethod
    def to_component_vector(X : torch.Tensor):
        """
        extract rotation, velocity, and position components from SE_2(3) batch of [N, 5, 5] tensors and return as [N, 9, 1] batched parameterization
        """

        C = X[:, 0:3, 0:3]
        v = X[:, 0:3, 3].unsqueeze(2)
        r = X[:, 0:3, 4].unsqueeze(2)

        phi = SO3.Log(C)

        vec = torch.cat((phi, v, r), dim=1)

        return vec
    
    @staticmethod
    def from_component_vector(X : torch.Tensor):
        """
        extract rotation, velocity, and position components from parameterized batch of shape [N, 9, 1] and return as batched tensor of SE_2(3) tensors
        of shape [N, 5, 5]
        """

        phi = X[:, 0:3, :]
        v = X[:, 3:6, :]
        r = X[:, 6:9, :]
        
        return SE23.from_components(SO3.Exp(phi), v, r)

    @staticmethod
    def wedge(xi: torch.Tensor):
        """
        Parameters
        ----------
        xi : torch.Tensor with shape (N, 9)
            batch of se_2(3) parameterizations to be projected to the lie algebra

        Returns
        -------
        torch.Tensor with shape (N, 5, 5)
            :math:`se_2(3)` matrix
        """

        # reduce dimensionality to desired degree
        # remove redundant dimensions
        if xi.dim() > 2:
            xi = torch.squeeze(xi, 2)

        xi_phi = xi[:, 0:3]
        xi_v = xi[:, 3:6].unsqueeze(2)
        xi_r = xi[:, 6:9].unsqueeze(2)

        Xi = torch.cat(
            (SO3.cross(xi_phi), xi_v, xi_r), dim=2
        )  # this yields a (N, 3, 5) matrix that must now be blocked with a (2, 5) batched matrix
        block = torch.cat((torch.zeros(xi.shape[0], 2, 3), batch_eye(xi.shape[0], 2, 2)), dim=2)

        return torch.cat((Xi, block), dim=1)

    @staticmethod
    def vee(Xi):
        Xi_phi = Xi[:, 0:3, 0:3]
        xi_phi = SO3.vee(Xi_phi)

        xi_v = Xi[:, 0:3, 3]
        xi_r = Xi[:, 0:3, 4]

        return torch.cat((xi_phi, xi_v, xi_r), dim=1)

    @staticmethod
    def exp(Xi):
        Xi_phi = Xi[:, 0:3, 0:3]
        xi_phi = SO3.vee(Xi_phi)
        xi_v = Xi[:, 0:3, 3]
        xi_r = Xi[:, 0:3, 4]
        C = SO3.exp(Xi_phi)

        J_left = SO3.left_jacobian(xi_phi)
        v = J_left @ xi_v
        r = J_left @ xi_r

        X = SE23.from_components(C, v, r)

        return X

    @staticmethod
    def log(X):
        (C, v, r) = SE23.to_components(X)
        phi = SO3.Log(C)
        J_left_inv = SO3.left_jacobian_inv(phi)
        v = J_left_inv @ v
        r = J_left_inv @ r

        Xi = torch.cat(
            (SO3.cross(phi), v, r), dim=2
        )  # this yields a (N, 3, 5) matrix that must now be blocked with a (2, 5) batched matrix
        block = torch.cat((torch.zeros(50, 2, 3), batch_eye(phi.shape[0], 2, 2)), dim=2)

        return torch.cat((Xi, block), dim=1)

    @staticmethod
    def adjoint(X):
        C, v, r = SE23.to_components(X)
        O = torch.zeros(v.shape[0], 3, 3)

        # creating block matrix
        b1 = torch.cat((C, O, O), dim=2)
        b2 = torch.cat((SO3.wedge(v) @ C, C, O), dim=2)
        b3 = torch.cat((SO3.wedge(r) @ C, O, C), dim=2)
        return torch.cat((b1, b2, b3), dim=1)

    @staticmethod
    def identity(N):
        return batch_eye(N, 5, 5)

    @staticmethod
    def left_jacobian(xi):
        xi_phi = xi[:, 0:3]
        xi_v = xi[:, 3:6]
        xi_r = xi[:, 6:9]

        J_left = batch_eye(xi.shape[0], 9, 9)

        small_angle_mask = is_close(
            torch.linalg.norm(xi_phi, dim=1), 0.0, SE23._small_angle_tol
        )
        # small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        # if np.linalg.norm(xi_phi) < SE23._small_angle_tol:
        #     return np.identity(9)

        # if small_angle_inds.shape[0] > 0 and small_angle_inds.numel():
        #     J_left[small_angle_inds] = batch_eye(small_angle_inds.shape[0], 9, 9)

        if large_angle_inds.shape[0] > 0 and large_angle_inds.numel():
            # filter only the "large" angle references
            xi_phi = xi_phi[large_angle_inds]
            xi_v = xi_v[large_angle_inds]
            xi_r = xi_r[large_angle_inds]

            Q_v = SE3._left_jacobian_Q_matrix(xi_phi, xi_v)
            Q_r = SE3._left_jacobian_Q_matrix(xi_phi, xi_r)
            J = SO3.left_jacobian(xi_phi)
            J_left[large_angle_inds, 0:3, 0:3] = J
            J_left[large_angle_inds, 3:6, 3:6] = J
            J_left[large_angle_inds, 6:9, 6:9] = J
            J_left[large_angle_inds, 3:6, 0:3] = Q_v
            J_left[large_angle_inds, 6:9, 0:3] = Q_r

        return J_left

    @staticmethod
    def left_jacobian_inv(xi):
        xi_phi = xi[:, 0:3]
        xi_v = xi[:, 3:6]
        xi_r = xi[:, 6:9]

        J_left = batch_eye(xi.shape[0], 9, 9)

        small_angle_mask = is_close(
            torch.linalg.norm(xi_phi, dim=1), 0.0, SE23._small_angle_tol
        )
        # small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        # if np.linalg.norm(xi_phi) < SE23._small_angle_tol:
        #     return np.identity(9)

        # if small_angle_inds.shape[0] > 0 and small_angle_inds.numel():
        #     J_left[small_angle_inds] = batch_eye(small_angle_inds.shape[0], 9, 9)

        if large_angle_inds.shape[0] > 0 and large_angle_inds.numel():
            # filter only the "large" angle references
            xi_phi = xi_phi[large_angle_inds]
            xi_v = xi_v[large_angle_inds]
            xi_r = xi_r[large_angle_inds]

            Q_v = SE3._left_jacobian_Q_matrix(xi_phi, xi_v)
            Q_r = SE3._left_jacobian_Q_matrix(xi_phi, xi_r)
            J = SO3.left_jacobian_inv(xi_phi)
            J_left[large_angle_inds, 0:3, 0:3] = J
            J_left[large_angle_inds, 3:6, 3:6] = J
            J_left[large_angle_inds, 6:9, 6:9] = J
            J_left[large_angle_inds, 3:6, 0:3] = -J @ Q_v @ J
            J_left[large_angle_inds, 6:9, 0:3] = -J @ Q_r @ J

        return J