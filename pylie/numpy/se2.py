from .base import MatrixLieGroup
import numpy as np
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
    def from_components(rot, trans):
        rot = np.array(rot)
        if rot.size == 3:
            rot = SO2.Exp(rot)
        C = rot.reshape((2,2))
        r = np.array(trans).reshape((-1,1))
        T = np.block([[C, r], [np.zeros((1, 2)), 1]])
        return T

    @staticmethod
    def to_components(T):
        C = T[0:2, 0:2]
        r = T[0:2, 2]
        return C, r

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
        return np.vstack((phi, xi_r.reshape((-1, 1))))

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
        xi_r = np.dot(SO2.left_jacobian_inv(SO2.vee(Xi_phi)), r.reshape((-1, 1)))
        Xi = np.block([[Xi_phi, xi_r], [np.zeros((1, 3))]])
        return Xi

    @staticmethod
    def odot(b):
        """
        odot operator as defined in Barfoot. I.e., an operator on an element of
        R^n such that

        a^wedge b = b^odot a
        """
        b = b.flatten()
        return np.block([[SO2.odot(b[0:2]), b[2]*np.identity(2)],
                         [np.zeros((1,1)), np.zeros((1,2))]])

    @staticmethod
    def left_jacobian(xi):
        xi = xi.flatten()
        rho = xi[1:]  # translation part
        phi = xi[0]  # rotation part

        # BELOW CODE IS FOR xi = [xi_r; phi]
        # So we have to switch it for the different ordering
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        phi_sq = phi * phi

        if phi_sq < 1e-15:
            A = 1 - 1.0 / 6.0 * phi_sq
            B = 0.5 * phi - 1.0 / 24.0 * phi * phi_sq
        else:
            A = sin_phi / phi
            B = (1 - cos_phi) / phi

        jac = np.zeros((SE2.dof, SE2.dof))
        jac[0][0] = A
        jac[0][1] = -B
        jac[1][0] = B
        jac[1][1] = A

        if phi_sq < 1e-15:
            jac[0][2] = rho[1] / 2.0 + phi * rho[0] / 6.0
            jac[1][2] = -rho[0] / 2.0 + phi * rho[1] / 6.0
        else:
            jac[0][2] = (
                rho[1] + phi * rho[0] - rho[1] * cos_phi - rho[0] * sin_phi
            ) / phi_sq
            jac[1][2] = (
                -rho[0] + phi * rho[1] + rho[0] * cos_phi - rho[1] * sin_phi
            ) / phi_sq

        jac[2][2] = 1

        # Jacobian order switched below to comply with our lie algebra ordering
        temp = np.array([jac[2, :], jac[0, :], jac[1, :]])
        temp2 = np.hstack(
            (
                temp[:, 2].reshape((-1, 1)),
                temp[:, 0].reshape((-1, 1)),
                temp[:, 1].reshape((-1, 1)),
            )
        )

        return temp2

    @staticmethod
    def adjoint(T):
        C = T[0:2, 0:2]
        r = T[0:2, 2].reshape((-1, 1))
        Om = np.array([[0, -1], [1, 0]])
        return np.block([[np.array([[1]]), np.zeros((1, 2))], [-np.dot(Om, r), C]])
