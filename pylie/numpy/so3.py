from .base import MatrixLieGroup
import numpy as np
from math import sin, cos, tan


class SO3(MatrixLieGroup):
    """
    An instantiation-free implementation of the SO3 matrix Lie group.
    """

    dof = 3

    @staticmethod
    def random():
        v = np.random.uniform(0, 2 * np.pi, (3, 1))
        return SO3.Exp(v)

    @staticmethod
    def wedge(xi):
        xi = xi.reshape((-1, 1))
        X = np.array(
            [
                [0, -xi[2, 0], xi[1, 0]],
                [xi[2, 0], 0, -xi[0, 0]],
                [-xi[1, 0], xi[0, 0], 0],
            ]
        )
        return X

    @staticmethod
    def vee(X):
        xi = np.array([[-X[1, 2]], [X[0, 2]], [-X[0, 1]]])
        return xi

    @staticmethod
    def exp(Xi):
        xi_vec = SO3.vee(Xi)
        phi = np.linalg.norm(xi_vec)
        if np.abs(phi) < SO3._small_angle_tol:
            return np.identity(3)
            
        a = xi_vec / phi
        X = (
            cos(phi) * np.identity(3)
            + (1 - np.cos(phi)) * np.dot(a, np.transpose(a))
            + sin(phi) * SO3.wedge(a)
        )
        return X

    @staticmethod
    def log(X):
        # The cosine of the rotation angle is related to the trace of C
        cos_angle = 0.5 * np.trace(X) - 0.5
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
        cos_angle = np.clip(cos_angle, -1., 1.)
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.isclose(angle, 0.):
            return (X - np.identity(3))

        # Otherwise take the matrix logarithm and return the rotation vector
        return (0.5 * angle / np.sin(angle)) * (X - np.transpose(X))

    @staticmethod
    def left_jacobian(xi):
        phi = np.linalg.norm(xi)
        if np.abs(phi) < SO3._small_angle_tol:
            return np.identity(3)

        a = xi / phi
        spp = sin(phi) / phi
        J = (
            spp * np.identity(3)
            + (1 - spp) * np.dot(a, np.transpose(a))
            + ((1 - cos(phi)) / phi) * SO3.wedge(a)
        )
        return J

    @staticmethod
    def left_jacobian_inv(xi):
        phi = np.linalg.norm(xi)
        if np.abs(phi) < SO3._small_angle_tol:
            return np.identity(3)
        a = xi / phi

        ct = 1 / tan(phi / 2)
        J_inv = (
            (phi / 2) * ct * np.identity(3)
            + (1 - phi / 2 * ct) * np.dot(a, np.transpose(a))
            - (phi / 2) * SO3.wedge(a)
        )
        return J_inv