from .base import MatrixLieGroup
import numpy as np


class SO2(MatrixLieGroup):
    """
    An instantiation-free implementation of the SO2 matrix Lie group.
    """

    dof = 1

    @staticmethod
    def random():
        phi = np.random.uniform(0, 2 * np.pi, (1, 1))
        return SO2.Exp(phi)

    @staticmethod
    def wedge(phi):
        if isinstance(phi, np.ndarray):
            if len(phi.shape) == 2:
                phi = phi[0, 0]
            elif len(phi.shape) == 1:
                phi = phi[0]
            else:
                raise RuntimeError("Input should be a scalar.")

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
    def left_jacobian(phi):
        # Near phi==0, use first order Taylor expansion
        if np.abs(phi) < SO2._small_angle_tol:
            return np.identity(2) + 0.5 * SO2.wedge(phi)

        s = np.sin(phi)
        c = np.cos(phi)

        return (s / phi) * np.identity(2) + ((1 - c) / phi) * SO2.wedge(1.0)

    @staticmethod
    def left_jacobian_inv(phi):
        # Near phi==0, use first order Taylor expansion
        if np.abs(phi) < SO2._small_angle_tol:
            return np.identity(2) - 0.5 * SO2.wedge(phi)

        half_angle = 0.5 * phi
        cot_half_angle = 1.0 / np.tan(half_angle)
        return half_angle * cot_half_angle * np.identity(2) - half_angle * SO2.wedge(
            1.0
        )

    @staticmethod
    def adjoint(C):
        return np.identity(2)

    @staticmethod 
    def odot(b):
        b = np.array(b).ravel()
        return np.array([-b[1], b[0]]).reshape((-1,1))