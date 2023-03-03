from .base import MatrixLieGroup, tonumpy
import jax.numpy as jnp
import numpy as onp
from jax import jit, lax
from .so3 import SO3

class SE3(MatrixLieGroup):
    """
    An instantiation-free implementation of the SE3 matrix Lie group.
    """

    dof = 6
    matrix_size = 4

    @staticmethod
    def random():
        return super(SE3, SE3).random()

    @staticmethod
    @tonumpy
    @jit
    def from_components(C, r):
        """
        Construct an SE(3) matrix from a rotation matrix and translation vector.
        """
        # Check if rotation component is a rotation vector or full DCM
        r = jnp.array(r).reshape((-1, 1))
        T = jnp.block([[C, r], [jnp.zeros((1, 3)), 1]])
        return T

    @staticmethod
    @tonumpy
    @jit
    def to_components(T):
        """
        Decompose an SE(3) matrix into a rotation matrix and translation vector.
        """
        C = T[0:3, 0:3]
        r = T[0:3, 3]
        return C, r

   

    @staticmethod
    @tonumpy
    @jit
    def wedge(xi):
        xi = jnp.array(xi).ravel()
        phi = xi[0:3]
        xi_r = xi[3:]
        Xi_phi = SO3.wedge(phi)
        Xi = jnp.block([[Xi_phi, xi_r.reshape((-1, 1))], [jnp.zeros((1, 4))]])
        return Xi

    @staticmethod
    @tonumpy
    @jit
    def vee(Xi):
        Xi_phi = Xi[0:3, 0:3]
        xi_r = Xi[0:3, 3]
        phi = SO3.vee(Xi_phi)
        return jnp.vstack((phi, xi_r.reshape((-1, 1))))

    @staticmethod
    @tonumpy
    @jit
    def exp(Xi):
        Xi_phi = Xi[0:3, 0:3]
        phi = SO3.vee(Xi_phi)
        xi_r = Xi[0:3, 3]
        C = SO3.exp(Xi_phi)
        r = jnp.dot(SO3.left_jacobian(phi), xi_r.reshape((-1, 1)))
        return jnp.block([[C, r], [jnp.zeros((1, 3)), 1]])

    @staticmethod
    @tonumpy
    @jit
    def Exp(x):
        phi =x[0:3]
        xi_r = x[3:]
        C = SO3.Exp(phi)
        r = jnp.dot(SO3.left_jacobian(phi), xi_r.reshape((-1, 1)))
        return jnp.block([[C, r], [jnp.zeros((1, 3)), 1]])

    @staticmethod
    @tonumpy
    @jit
    def log(T):
        Xi_phi = SO3.log(T[0:3, 0:3])
        r = T[0:3, 3]
        xi_r = jnp.dot(
            SO3.left_jacobian_inv(SO3.vee(Xi_phi)), r.reshape((-1, 1))
        )
        Xi = jnp.block([[Xi_phi, xi_r], [jnp.zeros((1, 4))]])
        return Xi

    @staticmethod
    @tonumpy
    @jit
    def Log(T):
        phi = SO3.Log(T[0:3, 0:3])
        r = T[0:3, 3]
        xi_r = jnp.dot(
            SO3.left_jacobian_inv(phi), r.reshape((-1, 1))
        )
        return jnp.vstack((phi, xi_r.reshape((-1, 1))))

    @staticmethod
    @tonumpy
    @jit
    def inverse(T):
        C_inv = T[0:3,0:3].transpose()
        r_inv = -jnp.dot(C_inv, T[0:3,3].reshape((-1,1)))
        return jnp.block([[C_inv, r_inv], [jnp.zeros((1, 3)), 1]])

    @staticmethod
    @tonumpy
    @jit
    def odot(b):
        b = jnp.array(b).ravel()
        X = jnp.zeros((4, 6))
        X = X.at[0:3, 0:3].set(SO3.odot(b[0:3]))
        X = X.at[0:3, 3:].set(b[3:]*onp.identity(3))
        # X = jnp.block([[SO3.odot(b[0:3]), b[3:]*onp.identity(3)], [onp.zeros((1, 6))]])
        return X

    @staticmethod
    @tonumpy
    @jit
    def adjoint(T):
        Ad = jnp.zeros((6, 6))
        Ad = Ad.at[0:3, 0:3].set(T[0:3, 0:3])
        Ad = Ad.at[3:6, 3:6].set(T[0:3, 0:3])
        Ad = Ad.at[3:6, 0:3].set(jnp.dot(SO3.wedge(T[0:3, 3]), T[0:3, 0:3]))
        return Ad

    @staticmethod
    @jit
    def _left_jacobian_Q_matrix(phi, rho):
        phi = jnp.array(phi).ravel()
        rho = jnp.array(rho).ravel()

        rx = SO3.wedge(rho)
        px = SO3.wedge(phi)

        ph = jnp.linalg.norm(phi)

        ph2 = ph * ph
        ph3 = ph2 * ph
        ph4 = ph3 * ph
        ph5 = ph4 * ph

        cph = jnp.cos(ph)
        sph = jnp.sin(ph)

        m1 = 0.5
        m2 = (ph - sph) / ph3
        m3 = (0.5 * ph2 + cph - 1.0) / ph4
        m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

        pxrx = px.dot(rx)
        rxpx = rx.dot(px)
        pxrxpx = pxrx.dot(px)

        t1 = rx
        t2 = pxrx + rxpx + pxrxpx
        t3 = px.dot(pxrx) + rxpx.dot(px) - 3.0 * pxrxpx
        t4 = pxrxpx.dot(px) + px.dot(pxrxpx)

        return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

    @staticmethod
    @tonumpy
    @jit
    def left_jacobian(xi):

        xi = jnp.array(xi).ravel()

        phi = xi[0:3]  # rotation part
        rho = xi[3:6]  # translation part

        return lax.cond(
            jnp.linalg.norm(phi) < SE3._small_angle_tol,
            lambda _: jnp.identity(6),
            lambda _: SE3._left_jacobian_large_angle(phi, rho),
            None
        )

    @staticmethod
    @jit
    def _left_jacobian_large_angle(phi, rho):
        Q = SE3._left_jacobian_Q_matrix(phi, rho)

        J = SO3.left_jacobian(phi)
        J_left = jnp.zeros((6,6))
        J_left = J_left.at[0:3, 0:3].set(J)
        J_left = J_left.at[3:6, 3:6].set(J)
        J_left = J_left.at[3:6, 0:3].set(Q)
        # out = jnp.block([[J, jnp.zeros((3,3))],[Q, J]])
        return J_left

    @staticmethod
    @tonumpy
    @jit
    def left_jacobian_inv(xi):
        xi = jnp.array(xi).ravel()

        phi = xi[0:3]  # rotation part
        rho = xi[3:6]  # translation part

        return lax.cond(
            jnp.linalg.norm(phi) < SE3._small_angle_tol,
            lambda _: jnp.identity(6),
            lambda _: SE3._left_jacobian_inv_large_angle(phi, rho),
        None)

    @staticmethod
    @jit
    def _left_jacobian_inv_large_angle(phi, rho):
        Q = SE3._left_jacobian_Q_matrix(phi, rho)

        J_inv = SO3.left_jacobian_inv(phi)
        J_left_inv = jnp.zeros((6,6))
        J_left_inv = J_left_inv.at[0:3, 0:3].set(J_inv)
        J_left_inv = J_left_inv.at[3:6, 3:6].set(J_inv)
        J_left_inv = J_left_inv.at[3:6, 0:3].set(-jnp.dot(J_inv, jnp.dot(Q, J_inv)))
        # out = jnp.block([[J_inv, jnp.zeros((3,3))],[-jnp.dot(J_inv, jnp.dot(Q, J_inv)), J_inv]])
        return J_left_inv

    @staticmethod
    def right_jacobian(xi):
        return SE3.left_jacobian(-xi)

    @staticmethod
    def right_jacobian_inv(xi):
        return SE3.left_jacobian_inv(-xi)
