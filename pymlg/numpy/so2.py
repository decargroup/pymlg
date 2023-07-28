from .base import MatrixLieGroup
import numpy as np


class SO2(MatrixLieGroup):
    """
    An instantiation-free implementation of the SO2 matrix Lie group.
    """

    dof = 1
    matrix_size = 2

    @staticmethod
    def random():

        phi = np.random.uniform(0, 2 * np.pi, (1, 1))
        return SO2.Exp(phi)

    @staticmethod
    def wedge(phi):
        phi = np.array(phi).item()

        X = np.array(
            [
                [0, -phi],
                [phi, 0],
            ]
        )
        return X

    @staticmethod
    def vee(X):
        phi = X[1, 0]
        return phi

    @staticmethod
    def exp(Xi):
        phi = SO2.vee(Xi)
        X = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        return X

    @staticmethod
    def log(X):
        phi = np.arctan2(X[1, 0], X[0, 0])
        return SO2.wedge(phi)

    @staticmethod
    def left_jacobian(x):
        return np.array([[1]])

    @staticmethod
    def left_jacobian_inv(x):
        return np.array([[1]])

    @staticmethod
    def right_jacobian(x):
        return np.array([[1]])

    @staticmethod
    def right_jacobian_inv(x):
        return np.array([[1]])

    @staticmethod
    def adjoint(C):
        return np.array([[1]])

    @staticmethod
    def odot(b):
        b = np.array(b).ravel()
        return np.array([-b[1], b[0]]).reshape((-1, 1))