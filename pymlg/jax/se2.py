from .base import MatrixLieGroup, fast_vector_norm
import jax.numpy as jnp
import numpy as np
from .so2 import SO2
from jax import jit, lax

class SE2(MatrixLieGroup):
    """
    An instantiation-free implementation of the SE2 matrix Lie group.
    """

    dof = 3
    matrix_size = 3

    @staticmethod
    @jit
    def from_components(C, r):
        """
        Construct an SE(2) matrix from a rotation matrix and translation vector.
        """
        C = C.reshape((2, 2))
        T = jnp.zeros((3, 3))
        T = T.at[0:2, 0:2].set(C)
        T = T.at[0:2, 2].set(r.ravel())
        T = T.at[2, 2].set(1)
        return T

    @staticmethod
    @jit
    def to_components(T):
        """
        Decompose an SE(2) matrix into a rotation matrix and translation vector.
        """
        C = T[0:2, 0:2]
        r = T[0:2, 2]
        return C, r

    @staticmethod
    @jit
    def wedge(xi):
        xi = jnp.array(xi).ravel()
        phi = xi[0]
        xi_r = xi[1:]
        Xi_phi = SO2.wedge(phi)
        Xi = jnp.zeros((3, 3))
        Xi = Xi.at[0:2, 0:2].set(Xi_phi)
        Xi = Xi.at[0:2, 2].set(xi_r)
        return Xi  

    @staticmethod
    @jit
    def vee(Xi):
        Xi_phi = Xi[0:2, 0:2]
        xi_r = Xi[0:2, 2]
        phi = SO2.vee(Xi_phi)
        return jnp.vstack((phi, xi_r.reshape((-1, 1))))

    @staticmethod
    @jit
    def exp(Xi):
        Xi_phi = Xi[0:2, 0:2]
        phi = SO2.vee(Xi_phi)
        xi_r = Xi[0:2, 2]
        C = SO2.exp(Xi_phi)
        r = jnp.dot(SE2.V_matrix(phi), xi_r.reshape((-1, 1)))
        return SE2.from_components(C, r)

    @staticmethod
    @jit
    def log(T):
        Xi_phi = SO2.log(T[0:2, 0:2])
        r = T[0:2, 2]
        xi_r = jnp.dot(SE2.V_matrix_inv(SO2.vee(Xi_phi)), r.reshape((-1, 1)))
        Xi = jnp.zeros((3, 3))
        Xi = Xi.at[0:2, 0:2].set(Xi_phi)
        Xi = Xi.at[0:2, 2].set(xi_r.ravel())
        return Xi

    @staticmethod
    @jit
    def Exp(x):
        phi = x[0]
        xi_r = x[1:]
        C = SO2.Exp(phi)
        r = jnp.dot(SE2.V_matrix(phi), xi_r.reshape((-1, 1)))
        return SE2.from_components(C, r)
    
    @staticmethod
    @jit
    def Log(T):
        phi = SO2.Log(T[0:2, 0:2])
        r = T[0:2, 2]
        xi_r = jnp.dot(SE2.V_matrix_inv(phi), r.reshape((-1, 1)))
        return jnp.vstack((phi, xi_r))

    @staticmethod
    @jit
    def odot(b):
        """
        odot operator as defined in Barfoot. I.e., an operator on an element of
        R^n such that

        a^wedge b = b^odot a
        """
        b = jnp.array(b).ravel()
        X = jnp.zeros((3, 3))
        X = X.at[0:2, 0].set(SO2.odot(b[0:2]).ravel())
        X = X.at[0:2, 1:3].set(b[2] * jnp.identity(2))
        return X

    @staticmethod
    @jit
    def left_jacobian(xi):
        xi = jnp.array(xi).ravel()
        rho = xi[1:]  # translation part
        phi = xi[0]  # rotation part

        # BELOW CODE IS FOR xi = [xi_r; phi]
        # So we have to switch it for the different ordering
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)
        phi_sq = phi * phi

        # if phi_sq < SE2._small_angle_tol:
        #     A = 1 - 1.0 / 6.0 * phi_sq
        #     B = 0.5 * phi - 1.0 / 24.0 * phi * phi_sq
        # else:
        #     A = sin_phi / phi
        #     B = (1 - cos_phi) / phi

        # fmt: off
        A, B = lax.cond(phi_sq < SE2._small_angle_tol,
                        lambda _: (1 - 1.0 / 6.0 * phi_sq, 0.5 * phi - 1.0 / 24.0 * phi * phi_sq),
                        lambda _: (sin_phi / phi, (1 - cos_phi) / phi),
                        operand=None)

        C, D = lax.cond(phi_sq < SE2._small_angle_tol,
                        lambda _: (rho[1] / 2.0 + phi * rho[0] / 6.0, -rho[0] / 2.0 + phi * rho[1] / 6.0),
                        lambda _: ((rho[1] + phi * rho[0] - rho[1] * cos_phi - rho[0] * sin_phi) / phi_sq,
                                   (-rho[0] + phi * rho[1] + rho[0] * cos_phi - rho[1] * sin_phi) / phi_sq),
                        operand=None)
        # fmt: on

        # if phi_sq < SE2._small_angle_tol:
        #     C = rho[1] / 2.0 + phi * rho[0] / 6.0
        #     D = -rho[0] / 2.0 + phi * rho[1] / 6.0
        # else:
        #     C = (
        #         rho[1] + phi * rho[0] - rho[1] * cos_phi - rho[0] * sin_phi
        #     ) / phi_sq
        #     D = (
        #         -rho[0] + phi * rho[1] + rho[0] * cos_phi - rho[1] * sin_phi
        #     ) / phi_sq

        # jac = jnp.zeros((SE2.dof, SE2.dof))
        # jac[0,0] = A
        # jac[0,1] = -B
        # jac[1,0] = B
        # jac[1,1] = A
        # jac[2,2] = 1
        # jac[0,2] = C 
        # jac[1,2] = D 

        jac = jnp.array([
            [A, -B, C],
            [B, A, D],
            [0, 0, 1],
            ])

        # Jacobian order switched below to comply with our lie algebra ordering
        temp = jnp.array([jac[2, :], jac[0, :], jac[1, :]])
        temp2 = jnp.hstack(
            (
                temp[:, 2].reshape((-1, 1)),
                temp[:, 0].reshape((-1, 1)),
                temp[:, 1].reshape((-1, 1)),
            )
        )

        return temp2

    @staticmethod
    @jit
    def adjoint(T):
        C = T[0:2, 0:2]
        r = T[0:2, 2].reshape((-1, 1))
        Om = jnp.array([[0, -1], [1, 0]])
        A = jnp.zeros((3, 3))
        A = A.at[0, 0].set(1)
        A = A.at[1:, 0].set(-jnp.dot(Om, r).ravel())
        A = A.at[1:, 1:].set(C)
        return A

    @staticmethod
    @jit
    def adjoint_algebra(Xi):
        A = jnp.zeros((3, 3))
        A = A.at[1, 0].set(Xi[1, 2])
        A = A.at[2, 0].set(-Xi[0, 2])
        A = A.at[1:, 1:].set(Xi[:2, :2])
        return A

    @staticmethod
    @jit
    def V_matrix(phi):
        # Near phi==0, use first order Taylor expansion
        phi = jnp.array(phi).reshape((1,))[0]
        s = jnp.sin(phi)
        c = jnp.cos(phi)

        return lax.cond(
            jnp.abs(phi) < SO2._small_angle_tol,
            lambda _: jnp.identity(2) + 0.5 * SO2.wedge(phi),
            lambda _: (s / phi) * jnp.identity(2) + ((1 - c) / phi) * SO2.wedge(1.0),
            operand=None,
        )

    @staticmethod
    @jit
    def V_matrix_inv(phi):
        # Near phi==0, use first order Taylor expansion
        phi = jnp.array(phi).reshape((1,))[0]
        half_angle = 0.5 * phi
        cot_half_angle = 1.0 / jnp.tan(half_angle)

        return lax.cond(
            jnp.abs(phi) < SO2._small_angle_tol,
            lambda _: jnp.identity(2) - 0.5 * SO2.wedge(phi),
            lambda _: half_angle * cot_half_angle * jnp.identity(2) - half_angle * SO2.wedge(1.0),
            operand=None,
        )