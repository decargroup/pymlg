from .base import MatrixLieGroup
import numpy as np
from math import sin, cos, tan
from .so2 import SO2


class SE2(MatrixLieGroup):
    """
    An instantiation-free implementation of the SE3 matrix Lie group.
    """

    dof = 3

    @staticmethod
    def random():
        phi = np.random.uniform(0, 2 * np.pi, (1, 1))
        r = np.random.normal(0, 1, (2, 1))
        C = SO2.Exp(phi)
        T = np.block([[C, r], [np.zeros((1, 2)), 1]])
        return T

    @staticmethod
    def wedge(xi):
        xi = xi.reshape((-1, 1))
        phi = xi[0:1, 0]
        xi_r = xi[1:, 0]
        Xi_phi = SO2.wedge(phi)
        return np.block([[Xi_phi, xi_r.reshape((-1, 1))], [np.zeros((1, 3))]])

    @staticmethod
    def vee(Xi):
        Xi_phi = Xi[0:2, 0:2]
        xi_r = Xi[0:2, 2]
        phi = SO2.vee(Xi_phi)
        return np.vstack((phi, xi_r.reshape((-1,1))))

    @staticmethod
    def exp(Xi):
        Xi_phi = Xi[0:2, 0:2]
        phi = SO2.vee(Xi_phi)
        xi_r = Xi[0:2, 2]
        C = SO2.exp(Xi_phi)
        r = np.dot(SO2.left_jacobian(phi), xi_r.reshape((-1, 1)))
        T = np.block([[C, r], [np.zeros((1, 2)), 1]])
        return T

    @staticmethod
    def log(T):
        Xi_phi = SO2.log(T[0:2, 0:2])
        r = T[0:2, 2]
        xi_r = np.dot(
            SO2.left_jacobian_inv(SO2.vee(Xi_phi)), r.reshape((-1, 1))
        )
        Xi = np.block([[Xi_phi, xi_r], [np.zeros((1, 3))]])
        return Xi
