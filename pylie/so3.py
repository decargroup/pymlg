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
        xi.reshape(-1, 1)
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
        a = xi_vec / phi
        X = (
            cos(phi) * np.identity(3)
            + (1 - np.cos(phi)) * np.dot(a, np.transpose(a))
            + sin(phi) * SO3.wedge(a)
        )
        return X

    @staticmethod
    def log(X):
        phi = np.arccos((np.trace(X) - 1) / 2)
        if np.abs(phi) < SO3._small_angle_tol:
            return np.identity(3)
        else:
            a = (1 / (2 * sin(phi))) * np.array(
                [[X[2, 1] - X[1, 2]], [X[0, 2] - X[2, 0]], [X[1, 0] - X[0, 1]]]
            )
            return phi * SO3.wedge(a)

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

if __name__ == "__main__":
    # Quick demo to show how to create an element
    phi = np.array([[0.1], [0.2], [0.3]])

    C = SO3.Exp(phi)

    print(C)
