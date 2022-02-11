from pdb import runeval
from .base import MatrixLieGroup
import torch

def bouter(vec1, vec2):
    """batch outer product"""
    return torch.einsum("bi, bj -> bij", vec1, vec2)

def batchtrace(mat):
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    tr = (torch.eye(mat.shape[1], dtype=mat.dtype, device=mat.device) * mat).sum(dim=1).sum(dim=1)

    return tr.view(mat.shape[0])


class SO3(MatrixLieGroup):
    """
    An torch instantiation-free implementation of the SO3 matrix Lie group.

    ALL functions are made to accept a batch: [N x ...].
    """

    dof = 3

    @staticmethod
    def random(N = 1):
        v = torch.rand((N, SO3.dof))
        return SO3.Exp(v)

    @staticmethod
    def wedge(phi):
        dim_batch = phi.shape[0]
        zero = phi.new_zeros(dim_batch)
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
    def vee(X):
        return torch.stack((X[:, 2, 1], X[:, 0, 2], X[:, 1, 0]), dim=1)

    # @staticmethod
    # def exp(Xi):
    #     xi_vec = SO3.vee(Xi)
    #     phi = np.linalg.norm(xi_vec)
    #     if np.abs(phi) < SO3._small_angle_tol:
    #         return np.identity(3)
            
    #     a = xi_vec / phi
    #     X = (
    #         cos(phi) * np.identity(3)
    #         + (1 - np.cos(phi)) * np.dot(a, np.transpose(a))
    #         + sin(phi) * SO3.wedge(a)
    #     )
    #     return X

    # @staticmethod
    # def log(X):
    #     phi = np.arccos((np.trace(X) - 1) / 2)
    #     if np.abs(phi) < SO3._small_angle_tol:
    #         return np.identity(3)
    #     else:
    #         a = (1 / (2 * sin(phi))) * np.array(
    #             [[X[2, 1] - X[1, 2]], [X[0, 2] - X[2, 0]], [X[1, 0] - X[0, 1]]]
    #         )
    #         return phi * SO3.wedge(a)

    @staticmethod
    def Exp(phi: torch.Tensor):
        """
        Exponential map of SO3. This function accepts a batch of rotation vectors
        in R^n of dimension [N x 3] and returns a batch of rotation matrices
        of dimension [N x 3 x 3].
        """
        if phi.shape == (3,1):
            phi = phi.flatten().unsqueeze(0)
        elif len(phi.shape) == 1:
            phi = phi.unsqueeze(0)
        elif len(phi.shape) == 2 and phi.shape[1] == 3:
            # acceptable.
            pass
        elif len(phi.shape) == 3 and (phi.shape[1] == 1 or phi.shape[2] == 1):
            phi = phi.squeeze()
        else:
            raise RuntimeError("Argument is not of acceptable dimensions.")

        angle = phi.norm(dim=1, keepdim=True)
        mask = angle[:, 0] < 1e-7
        dim_batch = phi.shape[0]
        Id = torch.eye(3, device=phi.device).expand(dim_batch, 3, 3)

        axis = phi[~mask] / angle[~mask]
        c = angle[~mask].cos().unsqueeze(2)
        s = angle[~mask].sin().unsqueeze(2)

        Rot = phi.new_empty(dim_batch, 3, 3)
        Rot[mask] = Id[mask] + SO3.wedge(phi[mask])
        Rot[~mask] = c * Id[~mask] + (1 - c) * bouter(axis, axis) + s * SO3.wedge(axis)
        return Rot

    @staticmethod
    def Log(C):
        """
        Logarithmic map of SO3. This function
        maps a batch of rotation matrices C [N x 3 x 3] to their corresponding
        elements in R^n. Output dimensions are [N x 3]
        """
        dim_batch = C.shape[0]
        Id = torch.eye(3, device = C.device).expand(dim_batch, 3, 3)

        cos_angle = (0.5 * batchtrace(C) - 0.5).clamp(-1.0+1e-7, 1.0-1e-7)
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding
        # errors
        angle = cos_angle.acos()
        mask = angle < 1e-14
        if mask.sum() == 0:
            angle = angle.unsqueeze(1).unsqueeze(1)
            return SO3.vee((0.5 * angle / angle.sin()) * (C - C.transpose(1, 2)))
        elif mask.sum() == dim_batch:
            # If angle is close to zero, use first-order Taylor expansion
            return SO3.vee(C - Id)
        phi = SO3.vee(C - Id)
        angle = angle
        phi[~mask] = SO3.vee(
            (0.5 * angle[~mask] / angle[~mask].sin()).unsqueeze(1).unsqueeze(2)
            * (C[~mask] - C[~mask].transpose(1, 2))
        )
        return phi

    # @staticmethod
    # def left_jacobian(xi):
    #     phi = np.linalg.norm(xi)
    #     if np.abs(phi) < SO3._small_angle_tol:
    #         return np.identity(3)

    #     a = xi / phi
    #     spp = sin(phi) / phi
    #     J = (
    #         spp * np.identity(3)
    #         + (1 - spp) * np.dot(a, np.transpose(a))
    #         + ((1 - cos(phi)) / phi) * SO3.wedge(a)
    #     )
    #     return J

    # @staticmethod
    # def left_jacobian_inv(xi):
    #     phi = np.linalg.norm(xi)
    #     if np.abs(phi) < SO3._small_angle_tol:
    #         return np.identity(3)
    #     a = xi / phi

    #     ct = 1 / tan(phi / 2)
    #     J_inv = (
    #         (phi / 2) * ct * np.identity(3)
    #         + (1 - phi / 2 * ct) * np.dot(a, np.transpose(a))
    #         - (phi / 2) * SO3.wedge(a)
    #     )
    #     return J_inv