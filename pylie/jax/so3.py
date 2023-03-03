from .base import MatrixLieGroup, tonumpy
import jax.numpy as np
from jax import  jit, lax
from functools import partial

class SO3(MatrixLieGroup):
    """
    An instantiation-free implementation of the SO3 matrix Lie group.
    """

    dof = 3
    matrix_size = 3

    @staticmethod
    @tonumpy
    @jit
    def random():
        return super(SO3, SO3).random()

    @staticmethod
    @tonumpy
    @jit
    def wedge(xi):
        xi = np.array(xi).ravel()
        X = np.array(
            [
                [0, -xi[2], xi[1]],
                [xi[2], 0, -xi[0]],
                [-xi[1], xi[0], 0],
            ]
        )
        return X

    @staticmethod
    @tonumpy
    @jit
    def cross(xi):
        """
        Alternate name for `SO3.wedge`
        """
        return SO3.wedge(xi)

    @staticmethod
    @tonumpy
    @jit
    def vee(X):
        return np.array([[-X[1, 2]], [X[0, 2]], [-X[0, 1]]])

    @staticmethod
    @tonumpy
    @jit
    def inverse(C):
        return C.transpose()


    @staticmethod
    @tonumpy
    @jit
    def exp(Xi):
        """
        Maps elements of the matrix Lie algebra so(3) to the group.

        From Section 8.3 of Lie Groups for Computer Vision by Ethan Eade. When
        theta is small, use Taylor series expansion given in Section 11.
        """
        return SO3.Exp(SO3.vee(Xi))
       

    @staticmethod
    @tonumpy
    @jit
    def Exp(xi):
        """
        Maps elements of the vector Lie algebra so(3) to the group.
        """
        phi = np.array(xi).ravel()
        angle = np.linalg.norm(phi)

        # If angle is close to zero, use first-order Taylor expansion
        out = lax.cond(
            angle < SO3._small_angle_tol,
            lambda _: np.eye(3) + SO3.wedge(phi),
            lambda _: SO3._Exp(phi, angle),
            None
        )
        return out
    
    @staticmethod
    def _Exp(phi, angle):
        A = np.sin(angle) / angle
        B = (1.0 - np.cos(angle)) / (angle**2)

        # Rodirgues rotation formula (103)
        Xi = SO3.wedge(phi)
        out = np.eye(3) + A * Xi + B * Xi @ Xi
        return out


    @staticmethod
    @tonumpy
    @jit
    def log(X):
        # The cosine of the rotation angle is related to the trace of C
        cos_angle = 0.5 * np.trace(X) - 0.5
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return lax.cond(
            angle < SO3._small_angle_tol,
            lambda _: X - np.eye(3),
            lambda _: (0.5 * angle / np.sin(angle)) * (X - np.transpose(X)),
            None
        )

    @staticmethod
    @tonumpy
    @jit
    def Log(C):
        """
        Maps elements of the matrix Lie group SO(3) to the vector Lie algebra
        so(3).
        """
        return SO3.vee(SO3.log(C))
        
    @staticmethod
    @tonumpy
    @jit
    def left_jacobian(xi):
        """
        Computes the Left Jacobian of SO(3).
        From Section 9.3 of Lie Groups for Computer Vision by Ethan Eade.  When
        angle is small, use Taylor series expansion given in Section 11.
        """
        xi = np.array(xi).ravel()
        angle = np.linalg.norm(xi)
        return lax.cond(
            angle < SO3._small_angle_tol,
            lambda _: SO3._left_jacobian_small_angle(xi, angle),
            lambda _: SO3._left_jacobian_large_angle(xi, angle),
            None
        )

    @staticmethod
    def _left_jacobian_small_angle(xi, angle):
        t2 = angle**2
        # Taylor series expansion.  See (157), (159).
        A = (1.0 / 2.0) * (
            1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0))
        )
        B = (1.0 / 6.0) * (
            1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0))
        )
        
        cross_xi = SO3.cross(xi)
        J_left = np.eye(3) + A * cross_xi + B * cross_xi @ cross_xi
        return J_left

    @staticmethod
    def _left_jacobian_large_angle(xi, angle):
        A = (1 - np.cos(angle)) / (angle**2)
        B = (angle - np.sin(angle)) / (angle**3)

        cross_xi = SO3.cross(xi)
        J_left = np.eye(3) + A * cross_xi + B * cross_xi @ cross_xi
        return J_left

    @staticmethod
    @tonumpy
    @jit
    def left_jacobian_inv(xi):
        """
        Computes the inverse of the left Jacobian of SO(3).
        From Section 9.3 of Lie Groups for Computer Vision by Ethan Eade. When
        angle is small, use Taylor series expansion given in Section 11.
        """
        xi = np.array(xi).ravel()
        angle = np.linalg.norm(xi)

        A = lax.cond(
            angle < SO3._small_angle_tol,
            lambda _: (1.0 / 12.0) * (
                1.0 + angle**2 / 60.0 * (1.0 + angle**2 / 42.0 * (1.0 + angle**2 / 40.0))
            ),
            lambda _: (1.0 / angle**2) * (
                1.0 - (angle * np.sin(angle) / (2.0 * (1.0 - np.cos(angle))))
            ),
            None
        )

        cross_xi = SO3.cross(xi)
        J_left_inv = np.eye(3) - 0.5 * cross_xi + A * cross_xi.dot(cross_xi)

        return J_left_inv

    @staticmethod
    @tonumpy
    def odot(xi):
        return -SO3.wedge(xi)

    @staticmethod
    def adjoint(C):
        return C

    @staticmethod
    @jit
    def identity():
        return np.identity(3)


