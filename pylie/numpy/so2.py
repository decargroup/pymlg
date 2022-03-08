from .base import MatrixLieGroup
import numpy as np
from math import sin, cos, tan, atan2


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
            phi = phi[0]

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
        X = np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])
        return X

    @staticmethod
    def log(X):
        phi = atan2(X[1, 0], X[0, 0])
        return SO2.wedge(phi)

    @staticmethod
    def left_jacobian(phi):
        # Near phi==0, use first order Taylor expansion
        if np.isclose(phi, 0.0):
            return np.identity(2) + 0.5 * SO2.wedge(phi)

        s = sin(phi)
        c = cos(phi)

        return (s / phi) * np.identity(2) + ((1 - c) / phi) * SO2.wedge(1.0)

    @staticmethod
    def left_jacobian_inv(phi):
        # Near phi==0, use first order Taylor expansion
        if np.isclose(phi, 0.0):
            return np.identity(2) - 0.5 * SO2.wedge(phi)

        half_angle = 0.5 * phi
        cot_half_angle = 1.0 / tan(half_angle)
        return half_angle * cot_half_angle * np.identity(2) - half_angle * SO2.wedge(
            1.0
        )
