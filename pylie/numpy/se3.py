from .base import MatrixLieGroup
import numpy as np
from math import sin, cos, tan
from .so3 import SO3


class SE3(MatrixLieGroup):
    """
    An instantiation-free implementation of the SE3 matrix Lie group.
    """

    dof = 6

    @staticmethod
    def synthesize(rot, disp):
        # Check if rotation component is a rotation vector or full DCM
        if rot.size == 9:
            C = rot
        else:
            C = SO3.Exp(rot)

        return np.block([[C, disp], [np.zeros((1, 3)), 1]])

    @staticmethod
    def random():
        phi = np.random.uniform(0, 2 * np.pi, (3, 1))
        r = np.random.normal(0, 1, (3, 1))
        C = SO3.Exp(phi)
        T = np.block([[C, r], [np.zeros((1, 3)), 1]])
        return T

    @staticmethod
    def wedge(xi):
        xi = xi.reshape((-1, 1))
        phi = xi[0:3, :]
        xi_r = xi[3:, :]
        Xi_phi = SO3.wedge(phi)
        return np.block([[Xi_phi, xi_r.reshape((-1, 1))], [np.zeros((1, 4))]])

    @staticmethod
    def vee(Xi):
        Xi_phi = Xi[0:3, 0:3]
        xi_r = Xi[0:3, 3]
        phi = SO3.vee(Xi_phi)
        return np.vstack((phi, xi_r.reshape((-1, 1))))

    @staticmethod
    def exp(Xi):
        Xi_phi = Xi[0:3, 0:3]
        phi = SO3.vee(Xi_phi)
        xi_r = Xi[0:3, 3]
        C = SO3.exp(Xi_phi)
        r = np.dot(SO3.left_jacobian(phi), xi_r.reshape((-1, 1)))
        T = np.block([[C, r], [np.zeros((1, 3)), 1]])
        return T

    @staticmethod
    def log(T):
        Xi_phi = SO3.log(T[0:3, 0:3])
        r = T[0:3, 3]
        xi_r = np.dot(SO3.left_jacobian_inv(SO3.vee(Xi_phi)), r.reshape((-1, 1)))
        Xi = np.block([[Xi_phi, xi_r], [np.zeros((1, 4))]])
        return Xi
