from .base import MatrixLieGroup
import numpy as np
from .so3 import SO3
from .se3 import SE3


class SE23(MatrixLieGroup):
    """
    An instantiation-free implementation of the SE_2(3) matrix Lie group.
    """

    dof = 9

    @staticmethod
    def synthesize(C, v, r):
        """
        Deprecated. Use `SE23.from_components(C, v, r)`.
        """
        return SE23.from_components(C, v, r)

    @staticmethod
    def decompose(element_SE23):
        """
        Deprecated. Use `SE23.to_components(X)`.
        """
        return SE23.to_components(element_SE23)

    @staticmethod
    def from_components(C, v, r):
        """
        Construct an :math:`SE_2(3)` matrix from attitude, velocity, position 
        components.

        Parameters
        ----------
        C : ndarray with shape (3,3)
            DCM/rotation matrix
        v : list[float] or ndarray with size 3
            velocity vector
        r : list[float] or ndarray with size 3
            position vector

        Returns
        -------
        ndarray with shape (5,5)
            :math:`SE_2(3)` matrix
        """
        v = np.array(v).reshape((-1,1))
        r = np.array(r).reshape((-1,1))
        return np.block([[C, v, r], [np.zeros((1, 3)), 1, 0], [np.zeros((1, 3)), 0, 1]])

    @staticmethod
    def to_components(X):
        """
        Extract rotation, velocity, position from SE_2(3) matrix.
        """
        C = X[0:3, 0:3]
        v = X[0:3, 3].reshape((-1,1))
        r = X[0:3, 4].reshape((-1,1))

        return (C, v, r)

    @staticmethod
    def random():
        phi = np.random.uniform(0, 2 * np.pi, (3, 1))
        v = np.random.normal(0, 1, (3, 1))
        r = np.random.normal(0, 1, (3, 1))

        C = SO3.Exp(phi)

        X = SE23.from_components(C, v, r)
        return X

    @staticmethod
    def wedge(xi):
        xi = np.array(xi).ravel()
        xi_phi = xi[0:3]
        xi_v = xi[3:6].reshape((-1,1))
        xi_r = xi[6:9].reshape((-1,1))
        Xi = np.block([[SO3.cross(xi_phi), xi_v, xi_r], [np.zeros((2, 5))]])
        return Xi

    @staticmethod
    def vee(Xi):
        Xi_phi = Xi[0:3, 0:3]
        xi_phi = SO3.vee(Xi_phi)

        xi_v = Xi[0:3, 3].reshape((-1,1))
        xi_r = Xi[0:3, 4].reshape((-1,1))

        return np.vstack((xi_phi, xi_v, xi_r))

    @staticmethod
    def exp(Xi):
        Xi_phi = Xi[0:3, 0:3]
        xi_phi = SO3.vee(Xi_phi)
        xi_v = Xi[0:3, 3].reshape((-1,1))
        xi_r = Xi[0:3, 4].reshape((-1,1))
        C = SO3.exp(Xi_phi)

        J_left = SO3.left_jacobian(xi_phi)
        v = np.dot(J_left, xi_v)
        r = np.dot(J_left, xi_r)

        X = np.block(
            [[C, v, r], [np.zeros((1, 3)), 1, 0], [np.zeros((1, 3)), 0, 1]]
        )

        return X

    @staticmethod
    def log(X):
        (C, v, r) = SE23.to_components(X)
        phi = SO3.Log(C)
        J_left_inv = SO3.left_jacobian_inv(phi)

        Xi = np.block(
            [
                [SO3.cross(phi), np.dot(J_left_inv, v), np.dot(J_left_inv, r)],
                [np.zeros((2, 5))],
            ]
        )

        return Xi

    @staticmethod
    def adjoint(X):
        C, v, r = SE23.to_components(X)
        O = np.zeros((3, 3))
        return np.block(
            [
                [C, O, O],
                [np.dot(SO3.wedge(v), C), C, O],
                [np.dot(SO3.wedge(r), C), O, C],
            ]
        )

    @staticmethod
    def identity():
        return np.eye(5)

        
    @staticmethod
    def left_jacobian(xi):
        xi_phi = xi[0:3]
        xi_v = xi[3:6].reshape((-1,1))
        xi_r = xi[6:9].reshape((-1,1))
        Q_v = SE3._left_jacobian_Q_matrix(xi_phi, xi_v)
        Q_r = SE3._left_jacobian_Q_matrix(xi_phi, xi_r)
        J = SO3.left_jacobian(xi_phi)
        J_left = np.identity(9)
        J_left[0:3, 0:3] = J
        J_left[3:6, 3:6] = J 
        J_left[6:9, 6:9] = J
        J_left[3:6, 0:3] = Q_v
        J_left[6:9, 0:3] = Q_r
        return J_left

    @staticmethod
    def left_jacobian_inv(xi):
        xi_phi = xi[0:3]
        xi_v = xi[3:6].reshape((-1,1))
        xi_r = xi[6:9].reshape((-1,1))
        Q_v = SE3._left_jacobian_Q_matrix(xi_phi, xi_v)
        Q_r = SE3._left_jacobian_Q_matrix(xi_phi, xi_r)
        J_inv = SO3.left_jacobian_inv(xi_phi)
        J_left_inv = np.identity(9)
        J_left_inv[0:3, 0:3] = J_inv
        J_left_inv[3:6, 3:6] = J_inv
        J_left_inv[6:9, 6:9] = J_inv
        J_left_inv[3:6, 0:3] = -J_inv @ Q_v @ J_inv
        J_left_inv[6:9, 0:3] = -J_inv @ Q_r @ J_inv
        return J_left_inv
