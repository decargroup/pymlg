from .base import MatrixLieGroupTorch
import torch
from .utils import *
from .so3 import SO3

# DISCLAIMER: this class is very much so un-tested

class SE3(MatrixLieGroupTorch):
    """
    An torch instantiation-free implementation of the SE3 matrix Lie group.

    ALL functions are made to accept a batch: [N x ...].
    """

    dof = 6
    matrix_size = 4
    
    @staticmethod
    def _left_jacobian_Q_matrix(xi_phi, xi_rho):

        rx = SO3.wedge(xi_rho)

        px = SO3.wedge(xi_phi)

        ph = xi_phi.norm(p=2, dim=1).view(-1)
        ph2 = ph * ph
        ph3 = ph2 * ph
        ph4 = ph3 * ph
        ph5 = ph4 * ph

        cph = ph.cos()
        sph = ph.sin()

        m1 = 0.5
        m2 = (ph - sph) / ph3
        m3 = (0.5 * ph2 + cph - 1.) / ph4
        m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

        m2 = m2.unsqueeze_(dim=1).unsqueeze_(dim=2).expand_as(rx)
        m3 = m3.unsqueeze_(dim=1).unsqueeze_(dim=2).expand_as(rx)
        m4 = m4.unsqueeze_(dim=1).unsqueeze_(dim=2).expand_as(rx)

        t1 = rx
        t2 = px.bmm(rx) + rx.bmm(px) + px.bmm(rx).bmm(px)
        t3 = px.bmm(px).bmm(rx) + rx.bmm(px).bmm(px) - 3. * px.bmm(rx).bmm(px)
        t4 = px.bmm(rx).bmm(px).bmm(px) + px.bmm(px).bmm(rx).bmm(px)

        Q = m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

        return Q.squeeze_()
    
    @staticmethod
    def random(N=1):
        """
        Generates a random batch of SE_(3) matricies.

        Parameters
        ----------
        N : int, optional
            batch size, by default 1

        Returns
        -------
        torch.Tensor with shape (N, 4, 4)
            batch of random :math:`SE(3)` matricies
        """
        
        phi = torch.rand(N, 3, 1)
        r = torch.rand(N, 3, 1)

        C = SO3.Exp(phi)

        return SE3.from_components(C, r)

    @staticmethod
    def from_components(C: torch.Tensor, r: torch.Tensor):
        """
        Construct a batch of :math:`SE(3)` matricies from attitude and position
        components.

        Parameters
        ----------
        C : torch.Tensor with shape (N, 3, 3)
            batch of rotation matricies/DCMs
        r : torch.Tensor with size (N, 3, 1)
            batch of position vectors

        Returns
        -------
        torch.Tensor with shape (N, 4, 4)
            :math:`SE(3)` matrix
        """

        # firstly, check that batch dimension for all 3 components matches
        if not (C.shape[0] == r.shape[0]):
            raise ValueError("Batch dimension for SE(3) components don't match.")

        X = batch_eye(C.shape[0], 4, 4)

        X[:, 0:3, 0:3] = C
        X[:, 0:3, 3] = r.squeeze(2)

        return X

    @staticmethod
    def to_components(X: torch.Tensor):
        """
        extract rotation, velocity, and position components from SE(3) batch
        """
        C = X[:, 0:3, 0:3]
        r = X[:, 0:3, 3]

        # enforce dimensionality
        return (C, r.unsqueeze(2))
    
    @staticmethod
    def wedge(xi: torch.Tensor):
        """
        Parameters
        ----------
        xi : torch.Tensor with shape (N, 6)
            batch of se(3) parameterizations to be projected to the lie algebra

        Returns
        -------
        torch.Tensor with shape (N, 4, 4)
            :math:`se(3)` matrix
        """

        # reduce dimensionality to desired degree
        # remove redundant dimensions
        if xi.dim() > 2:
            xi = torch.squeeze(xi, 2)

        xi_phi = xi[:, 0:3]
        xi_r = xi[:, 3:6].unsqueeze(2)

        Xi = torch.cat(
            (SO3.cross(xi_phi), xi_r), dim=2
        )  # this yields a (N, 3, 4) matrix that must now be blocked with a (1, 4) batched matrix

        # generating a (N, 1, 4) batched matrix to append
        b1 = torch.tensor([0, 0, 0, 0]).reshape(1, 1, 4)
        block = b1.repeat(Xi.shape[0], 1, 1)

        return torch.cat((Xi, block), dim=1)

    @staticmethod
    def vee(Xi : torch.Tensor):
        Xi_phi = Xi[:, 0:3, 0:3]
        xi_phi = SO3.vee(Xi_phi)

        xi_r = Xi[:, 0:3, 3].unsqueeze(2)

        return torch.cat((xi_phi, xi_r), dim=1)

    @staticmethod
    def exp(Xi : torch.Tensor):
        Xi_phi = Xi[:, 0:3, 0:3]
        xi_phi = SO3.vee(Xi_phi)
        xi_r = Xi[:, 0:3, 3].unsqueeze(2)
        C = SO3.exp(Xi_phi)

        J_left = SO3.left_jacobian(xi_phi)
        r = J_left @ xi_r

        X = SE3.from_components(C, r)

        return X

    @staticmethod
    def log(X : torch.Tensor):
        (C, r) = SE3.to_components(X)
        phi = SO3.Log(C)
        J_left_inv = SO3.left_jacobian_inv(phi)
        r = J_left_inv @ r

        Xi = torch.cat(
            (SO3.cross(phi), r), dim=2
        )  # this yields a (N, 3, 4) matrix that must now be blocked with a (1, 4) batched matrix

        # generating a (N, 1, 4) batched matrix to append
        b1 = torch.tensor([0, 0, 0, 0]).reshape(1, 1, 4)
        block = b1.repeat(Xi.shape[0], 1, 1)

        return torch.cat((Xi, block), dim=1)
    
    @staticmethod
    def odot(b : torch.Tensor):
        X = torch.zeros(b.shape[0], 4, 6)
        X[:, 0:3, 0:3] = SO3.odot(b[0:3])
        X[:, 0:3, 3:6] = b[:, 3] * batch_eye(b.shape[0], 3, 3)
        return X

    @staticmethod
    def adjoint(X):
        C, r = SE3.to_components(X)
        O = torch.zeros(r.shape[0], 3, 3)

        # creating block matrix
        b1 = torch.cat((C, O), dim=2)
        b3 = torch.cat((SO3.wedge(r) @ C, C), dim=2)
        return torch.cat((b1, b3), dim=1)
    
    @staticmethod
    def adjoint_algebra(Xi):
        A = torch.zeros(Xi.shape[0], 6, 6)
        A[:, 0:3, 0:3] = Xi[:, 0:3, 0:3]
        A[:, 3:6, 0:3] = SO3.wedge(Xi[:, 0:3, 3])
        A[:, 3:6, 3:6] = Xi[:, 0:3, 0:3]
        return A

    @staticmethod
    def identity(N=1):
        return batch_eye(N, 4, 4)

    @staticmethod
    def left_jacobian(xi):
        xi_phi = xi[:, 0:3]
        xi_r = xi[:, 3:6]

        J_left = batch_eye(xi.shape[0], 6, 6)

        small_angle_mask = is_close(
            torch.linalg.norm(xi_phi, dim=1), 0.0, SE3._small_angle_tol
        )
        # small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=True)[0]

        if large_angle_inds.numel():
            # filter only the "large" angle references
            xi_phi = xi_phi[large_angle_inds]
            xi_r = xi_r[large_angle_inds]

            Q_r = SE3._left_jacobian_Q_matrix(xi_phi, xi_r)
            J = SO3.left_jacobian(xi_phi)
            J_left[large_angle_inds, 0:3, 0:3] = J
            J_left[large_angle_inds, 3:6, 3:6] = J
            J_left[large_angle_inds, 3:6, 0:3] = Q_r

        return J_left
    
    @staticmethod
    def inverse(X):
        C, r = SE3.to_components(X)
        C_inv = C.transpose(1, 2)
        r_inv = -C_inv @ r

        X_inv = SE3.from_components(C_inv, r_inv)

        return X_inv

    @staticmethod
    def left_jacobian_inv(xi):
        xi_phi = xi[:, 0:3]
        xi_r = xi[:, 3:6]

        J_left = batch_eye(xi.shape[0], 6, 6)

        small_angle_mask = is_close(
            torch.linalg.norm(xi_phi, dim=1), 0.0, SE3._small_angle_tol
        )
        # small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=True)[0]

        if large_angle_inds.shape[0] > 0 and large_angle_inds.numel():
            # filter only the "large" angle references
            xi_phi = xi_phi[large_angle_inds]
            xi_r = xi_r[large_angle_inds]

            Q_r = SE3._left_jacobian_Q_matrix(xi_phi, xi_r)
            J = SO3.left_jacobian_inv(xi_phi)
            J_left[large_angle_inds, 0:3, 0:3] = J
            J_left[large_angle_inds, 3:6, 3:6] = J
            J_left[large_angle_inds, 3:6, 0:3] = -J @ Q_r @ J

        return J_left