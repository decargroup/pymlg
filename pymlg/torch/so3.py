from .base import MatrixLieGroupTorch
import torch
import numpy as np
from .utils import *

# DISCLAIMER: this class is very much so un-tested


def bouter(vec1, vec2):
    """batch outer product"""
    a = torch.einsum("bik, bjk -> bij", vec1, vec2)
    return a


def batchtrace(mat):
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    tr = (
        (torch.eye(mat.shape[1], dtype=mat.dtype, device=mat.device) * mat)
        .sum(dim=1)
        .sum(dim=1)
    )

    return tr.view(mat.shape[0])


class SO3(MatrixLieGroupTorch):
    """
    An torch instantiation-free implementation of the SO3 matrix Lie group.

    ALL functions are made to accept a batch: [N x ...].
    """

    dof = 3
    matrix_size = 3

    @staticmethod
    def random(N=1):
        v = torch.rand((N, SO3.dof))
        return SO3.Exp(v)

    @staticmethod
    def wedge(phi):
        # protect against empty tensor
        if not phi.numel():
            phi = torch.zeros(1, 3, device=phi.device)
        # remove redundant dimensions
        elif phi.dim() > 2:
            phi = torch.squeeze(phi, 2)
        dim_batch = phi.shape[0]
        zero = phi.new_zeros(dim_batch, device=phi.device)
        return torch.stack(
            (
                zero,
                -phi[:, 2],
                phi[:, 1],
                phi[:, 2],
                zero,
                -phi[:, 0],
                -phi[:, 1],
                phi[:, 0],
                zero,
            ),
            1,
        ).view(dim_batch, 3, 3)

    @staticmethod
    def cross(xi):
        """
        Alternate name for `SO3.wedge`
        """
        return SO3.wedge(xi)

    @staticmethod
    def vee(X):
        return torch.stack((X[:, 2, 1], X[:, 0, 2], X[:, 1, 0]), dim=1).unsqueeze(2)

    @staticmethod
    def Exp(phi: torch.Tensor):
        """
        Exponential map of SO3. This function accepts a batch of rotation vectors
        in R^n of dimension [N x 3 x 1] and returns a batch of rotation matrices
        of dimension [N x 3 x 3].
        """
        if phi.shape == (3, 1):
            phi = phi.unsqueeze(0)
        elif len(phi.shape) == 1:
            phi = phi.view(1, 3, 1)
        elif len(phi.shape) == 2 and phi.shape[1] == 3:
            phi = phi.unsqueeze(2)
        elif len(phi.shape) == 3 and (phi.shape[1] == 3):
            pass  # acceptable
        else:
            raise RuntimeError("Argument is not of acceptable dimensions.")

        # catch all fall-through errors
        if (phi.shape[1] != 3) and ((phi.shape[2] != 1)):
            raise RuntimeError(
                "phi argument in SO3 Exponential is not of acceptable dimension."
            )

        angle = phi.norm(dim=1, keepdim=True)
        mask = angle[:, 0, 0] < 1e-7
        mask = mask.to(phi.device)
        dim_batch = phi.shape[0]
        Id = torch.eye(3, device=phi.device).expand(dim_batch, 3, 3)

        axis = phi[~mask] / angle[~mask]
        c = angle[~mask].cos()
        s = angle[~mask].sin()

        Rot = phi.new_empty(dim_batch, 3, 3)
        Rot[mask] = Id[mask] + SO3.wedge(phi[mask])
        Rot[~mask] = c * Id[~mask] + (1 - c) * bouter(axis, axis) + s * SO3.wedge(axis)

        return Rot
    
    @staticmethod
    def odot(phi):
        return -SO3.wedge(phi)

    @staticmethod
    def to_euler(C, order="123"):
        """
        Convert a batch of rotation matrices to a batch of (1, 2, 3) euler angles
        """

        # if C.shape[0] != 1:
        #     raise ValueError(
        #         "Only batch size 1 is supported, currently - I will eventually make a mask for near gimbal-lock angles."
        #     )

        if order == "123":
            theta_1 = torch.arcsin(C[:, 2, 0])
            yaw_1 = torch.empty(C.shape[0], device=C.device)
            roll_1 = torch.empty(C.shape[0], device=C.device)

            gimbal_lock_mask = torch.logical_or(
                torch.isclose(theta_1, torch.Tensor([torch.pi / 2])),
                torch.isclose(theta_1, torch.Tensor([-torch.pi / 2])),
            )

            non_gimbal_lock_mask = torch.logical_not(gimbal_lock_mask)

            yaw_1[gimbal_lock_mask] = torch.zeros(1, device=C.device)
            roll_1[gimbal_lock_mask] = torch.arctan2(
                C[gimbal_lock_mask, 1, 2], C[gimbal_lock_mask, 1, 1]
            )

            roll_1[non_gimbal_lock_mask] = torch.arctan2(
                -C[non_gimbal_lock_mask, 2, 1]
                / torch.cos(theta_1[non_gimbal_lock_mask]),
                C[non_gimbal_lock_mask, 2, 2]
                / torch.cos(theta_1[non_gimbal_lock_mask]),
            )

            yaw_1[non_gimbal_lock_mask] = torch.arctan2(
                -C[non_gimbal_lock_mask, 1, 0]
                / torch.cos(theta_1[non_gimbal_lock_mask]),
                C[non_gimbal_lock_mask, 0, 0]
                / torch.cos(theta_1[non_gimbal_lock_mask]),
            )

            # if (torch.isclose(theta_1, torch.Tensor([torch.pi / 2]))) or (
            #     torch.isclose(theta_1, torch.Tensor([-torch.pi / 2]))
            # ):
            #     yaw_1 = torch.zeros(1, device=C.device)
            #     roll_1 = torch.arctan2(C[:, 1, 2], C[:, 1, 1])
            # else:
            #     roll_1 = torch.arctan2(
            #         -C[:, 2, 1] / torch.cos(theta_1), C[:, 2, 2] / torch.cos(theta_1)
            #     )
            #     yaw_1 = torch.arctan2(
            #         -C[:, 1, 0] / torch.cos(theta_1), C[:, 0, 0] / torch.cos(theta_1)
            #     )

            euler_1 = torch.cat(
                (roll_1.unsqueeze(1), theta_1.unsqueeze(1), yaw_1.unsqueeze(1)), dim=1
            )

            return euler_1
        else:
            raise NotImplementedError("Only 123 euler angle order is supported.")

    @staticmethod
    def log(C):
        """
        Logarithmic map of SO3. This function
        maps a batch of rotation matrices C [N x 3 x 3] to their corresponding lie algebra elements in R^(3x3) [N x 3 x 3].
        """
        dim_batch = C.shape[0]
        Id = torch.eye(3, device=C.device).expand(dim_batch, 3, 3)

        cos_angle = (0.5 * batchtrace(C) - 0.5).clamp(-1.0 + 1e-10, 1.0 - 1e-10)

        # assert non-nan angle
        assert not torch.isnan(cos_angle).any(), "NaN angle detected in SO3.Log"
        
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding
        # errors
        angle = cos_angle.acos()

        mask = angle < 1e-14

        # firstly, check if batch is singular element (pi approxiation currently only implemented for batch size 1)
        if C.shape[0] == 1:

            # check if angle is close to pi
            if np.isclose(angle.__float__(), np.pi, atol=1e-9):
                # if so, return a manually generated rotation vector that protects
                # against formulaic failure around pi
                phi_constructed = torch.Tensor([C[:, 0, 2], C[:, 1, 2], 1 + C[:, 2, 2]])
                rho = 1 / torch.sqrt(2 * (1 + C[0, 2, 2]))
                phi_constructed = rho * phi_constructed
                return SO3.wedge((torch.pi * phi_constructed).reshape(1, 3, 1))

        if mask.sum() == 0:
            angle = angle.unsqueeze(1).unsqueeze(1)
            return (0.5 * angle / angle.sin()) * (C - C.transpose(1, 2))
        elif mask.sum() == dim_batch:
            # If angle is close to zero, use first-order Taylor expansion
            return (C - Id)
        phi = SO3.vee(C - Id)
        angle = angle
        phi[~mask] = SO3.vee(
            (0.5 * angle[~mask] / angle[~mask].sin()).unsqueeze(1).unsqueeze(2)
            * (C[~mask] - C[~mask].transpose(1, 2))
        )
        return SO3.wedge(phi)

    @staticmethod
    def A_lj(t_norm, small=True):
        t2 = t_norm**2
        if small:
            return (1.0 / 2.0) * (
                1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0))
            )
        else:
            return (1 - torch.cos(t_norm)) / (t_norm**2)

    @staticmethod
    def B_lj(t_norm, small=True):
        t2 = t_norm**2
        if small:
            return (1.0 / 6.0) * (
                1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0))
            )
        else:
            return (t_norm - torch.sin(t_norm)) / (t_norm**3)

    @staticmethod
    def A_inv_lj(t_norm, small=True):
        if small:
            t2 = t_norm**2
            return (1.0 / 12.0) * (
                1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0))
            )
        else:
            return (1.0 / t_norm**2) * (
                1.0 - (t_norm * torch.sin(t_norm) / (2.0 * (1.0 - torch.cos(t_norm))))
            )

    @staticmethod
    def left_jacobian(xi):
        """
        Computes the Left Jacobian of SO(3).
        From Section 9.3 of Lie Groups for Computer Vision by Ethan Eade.  When
        angle is small, use Taylor series expansion given in Section 11.
        """

        xi_norm = torch.linalg.norm(xi, dim=1)

        small_angle_mask = is_close(xi_norm, 0.0)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=True)[0]
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=True)[0]

        J_left = torch.empty(xi.shape[0], 3, 3, dtype=xi.dtype)

        cross_xi = SO3.wedge(xi)

        if small_angle_inds.numel():
            J_left[small_angle_inds] = (
                torch.eye(3, 3).expand(small_angle_inds.shape[0], 3, 3)
                + SO3.A_lj(xi_norm[small_angle_inds], small=True)
                .reshape(-1, 1)
                .unsqueeze(2)
                * cross_xi[small_angle_inds]
                + SO3.B_lj(xi_norm[small_angle_inds], small=True)
                .reshape(-1, 1)
                .unsqueeze(2)
                * torch.bmm(cross_xi[small_angle_inds], cross_xi[small_angle_inds])
            )
        if large_angle_inds.numel():
            J_left[large_angle_inds] = (
                torch.eye(3, 3).expand(large_angle_inds.shape[0], 3, 3)
                + SO3.A_lj(xi_norm[large_angle_inds], small=False)
                .reshape(-1, 1)
                .unsqueeze(2)
                * cross_xi[large_angle_inds]
                + SO3.B_lj(xi_norm[large_angle_inds], small=False)
                .reshape(-1, 1)
                .unsqueeze(2)
                * torch.bmm(cross_xi[large_angle_inds], cross_xi[large_angle_inds])
            )

        return J_left

    @staticmethod
    def left_jacobian_inv(xi):
        """
        Computes the inverse of the left Jacobian of SO(3).
        From Section 9.3 of Lie Groups for Computer Vision by Ethan Eade. When
        angle is small, use Taylor series expansion given in Section 11.
        """

        xi_norm = torch.linalg.norm(xi, dim=1)

        small_angle_mask = is_close(xi_norm, 0.0, tol=SO3._small_angle_tol)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=True)[0]
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=True)[0]

        J_left = torch.empty(xi.shape[0], 3, 3, dtype=xi.dtype)

        cross_xi = SO3.wedge(xi)

        if small_angle_inds.numel():
            J_left[small_angle_inds] = (
                batch_eye(small_angle_inds.shape[0], 3, 3)
                - 0.5 * cross_xi[small_angle_inds]
                + SO3.A_inv_lj(xi_norm[small_angle_inds], small=True)
                .reshape(-1, 1)
                .unsqueeze(2)
                * torch.bmm(cross_xi[small_angle_inds], cross_xi[small_angle_inds])
            )
        if large_angle_inds.numel():
            J_left[large_angle_inds] = (
                batch_eye(large_angle_inds.shape[0], 3, 3)
                - 0.5 * cross_xi[large_angle_inds]
                + SO3.A_inv_lj(xi_norm[large_angle_inds], small=False)
                .reshape(-1, 1)
                .unsqueeze(2)
                * torch.bmm(cross_xi[large_angle_inds], cross_xi[large_angle_inds])
            )

        return J_left
    
    @staticmethod
    def adjoint(C):
        return C

    @staticmethod
    def adjoint_algebra(Xi):
        return Xi
    
    @staticmethod
    def to_quat(C, order='wxyz'):
        """Convert a rotation matrix to a unit length quaternion.

            Valid orders are 'xyzw' and 'wxyz'.
        """
        if C.dim() < 3:
            C = C.unsqueeze(dim=0)

        qw = 0.5 * torch.sqrt(1. + C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2])
        qx = qw.new_empty(qw.shape)
        qy = qw.new_empty(qw.shape)
        qz = qw.new_empty(qw.shape)

        near_zero_mask = (qw).abs_().lt(1e-6)

        if sum(near_zero_mask) > 0:
            cond1_mask = near_zero_mask & \
                (C[:, 0, 0] > C[:, 1, 1]).squeeze_() & \
                (C[:, 0, 0] > C[:, 2, 2]).squeeze_()
            cond1_inds = cond1_mask.nonzero(as_tuple=False).squeeze_(dim=1)

            if len(cond1_inds) > 0:
                R_cond1 = C[cond1_inds]
                d = 2. * np.sqrt(1. + R_cond1[:, 0, 0] -
                                    R_cond1[:, 1, 1] - R_cond1[:, 2, 2])
                qw[cond1_inds] = (R_cond1[:, 2, 1] - R_cond1[:, 1, 2]) / d
                qx[cond1_inds] = 0.25 * d
                qy[cond1_inds] = (R_cond1[:, 1, 0] + R_cond1[:, 0, 1]) / d
                qz[cond1_inds] = (R_cond1[:, 0, 2] + R_cond1[:, 2, 0]) / d

            cond2_mask = near_zero_mask & (C[:, 1, 1] > C[:, 2, 2]).squeeze_()
            cond2_inds = cond2_mask.nonzero(as_tuple=False).squeeze_(dim=1)

            if len(cond2_inds) > 0:
                R_cond2 = C[cond2_inds]
                d = 2. * np.sqrt(1. + R_cond2[:, 1, 1] -
                                    R_cond2[:, 0, 0] - R_cond2[:, 2, 2])
                qw[cond2_inds] = (R_cond2[:, 0, 2] - R_cond2[:, 2, 0]) / d
                qx[cond2_inds] = (R_cond2[:, 1, 0] + R_cond2[:, 0, 1]) / d
                qy[cond2_inds] = 0.25 * d
                qz[cond2_inds] = (R_cond2[:, 2, 1] + R_cond2[:, 1, 2]) / d

            cond3_mask = near_zero_mask & cond1_mask.logical_not() & cond2_mask.logical_not()
            cond3_inds = cond3_mask.nonzero(as_tuple=False).squeeze_(dim=1)

            if len(cond3_inds) > 0:
                R_cond3 = C[cond3_inds]
                d = 2. * \
                    np.sqrt(1. + R_cond3[:, 2, 2] -
                            R_cond3[:, 0, 0] - R_cond3[:, 1, 1])
                qw[cond3_inds] = (R_cond3[:, 1, 0] - R_cond3[:, 0, 1]) / d
                qx[cond3_inds] = (R_cond3[:, 0, 2] + R_cond3[:, 2, 0]) / d
                qy[cond3_inds] = (R_cond3[:, 2, 1] + R_cond3[:, 1, 2]) / d
                qz[cond3_inds] = 0.25 * d

        far_zero_mask = near_zero_mask.logical_not()
        far_zero_inds = far_zero_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        if len(far_zero_inds) > 0:
            R_fz = C[far_zero_inds]
            d = 4. * qw[far_zero_inds]
            qx[far_zero_inds] = (R_fz[:, 2, 1] - R_fz[:, 1, 2]) / d
            qy[far_zero_inds] = (R_fz[:, 0, 2] - R_fz[:, 2, 0]) / d
            qz[far_zero_inds] = (R_fz[:, 1, 0] - R_fz[:, 0, 1]) / d

        # Check order last
        if order == 'xyzw':
            quat = torch.cat([qx.unsqueeze_(dim=1),
                                qy.unsqueeze_(dim=1),
                                qz.unsqueeze_(dim=1),
                                qw.unsqueeze_(dim=1)], dim=1).squeeze_()
        elif order == 'wxyz':
            quat = torch.cat([qw.unsqueeze_(dim=1),
                                qx.unsqueeze_(dim=1),
                                qy.unsqueeze_(dim=1),
                                qz.unsqueeze_(dim=1)], dim=1).squeeze_()
        else:
            raise ValueError(
                "Valid orders are 'xyzw' and 'wxyz'. Got '{}'.".format(order))

        return quat

    @staticmethod
    def from_quat(quat, order="wxyz"):
        """Form a rotation matrix from a unit length quaternion.
        Valid orders are 'xyzw' and 'wxyz'.
        from https://github.com/utiasSTARS/liegroups/blob/fe1d376b7d33809dec78724b456f01833507c305/liegroups/torch/so3.py#L60
        """
        if quat.dim() < 2:
            quat = quat.unsqueeze(dim=0)

        if not torch.all(is_close(quat.norm(p=2, dim=1), 1.0)):
            raise ValueError("Quaternions must be unit length")

        if order == "xyzw":
            qx = quat[:, 0]
            qy = quat[:, 1]
            qz = quat[:, 2]
            qw = quat[:, 3]
        elif order == "wxyz":
            qw = quat[:, 0]
            qx = quat[:, 1]
            qy = quat[:, 2]
            qz = quat[:, 3]
        else:
            raise ValueError(
                "Valid orders are 'xyzw' and 'wxyz'. Got '{}'.".format(order)
            )

        # Form the matrix
        mat = quat.new_empty(quat.shape[0], 3, 3)

        qw2 = qw * qw
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        mat[:, 0, 0] = 1.0 - 2.0 * (qy2 + qz2)
        mat[:, 0, 1] = 2.0 * (qx * qy - qw * qz)
        mat[:, 0, 2] = 2.0 * (qw * qy + qx * qz)

        mat[:, 1, 0] = 2.0 * (qw * qz + qx * qy)
        mat[:, 1, 1] = 1.0 - 2.0 * (qx2 + qz2)
        mat[:, 1, 2] = 2.0 * (qy * qz - qw * qx)

        mat[:, 2, 0] = 2.0 * (qx * qz - qw * qy)
        mat[:, 2, 1] = 2.0 * (qw * qx + qy * qz)
        mat[:, 2, 2] = 1.0 - 2.0 * (qx2 + qy2)

        return mat
