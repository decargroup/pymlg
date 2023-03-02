from .base import MatrixLieGroup, fast_vector_norm
import jax.numpy as np
from jax import random, jit, lax
from .so3 import SO3

try:
    # We do not want to make ROS a hard dependency, so we import it only if
    # available.
    from geometry_msgs.msg import Pose
except ImportError:
    pass  # ROS is not installed
except:
    raise

key = random.PRNGKey(0)
class SE3(MatrixLieGroup):
    """
    An instantiation-free implementation of the SE3 matrix Lie group.
    """

    dof = 6
    matrix_size = 4
    _identity = None

    # @staticmethod
    # def identity():
    #     if SE3._identity is None:
    #         SE3._identity = np.eye(4)
    #     return SE3._identity

    @staticmethod
    @jit
    def synthesize(C, r):
        """
        Deprecated. Use `SE3.from_components(C,r)` instead.

        Construct an SE(3) matrix from a rotation matrix and translation vector.
        """
        return SE3.from_components(C, r)

    @staticmethod
    @jit
    def from_components(C, r):
        """
        Construct an SE(3) matrix from a rotation matrix and translation vector.
        """
        # Check if rotation component is a rotation vector or full DCM
        r = np.array(r).reshape((-1, 1))
        T = np.block([[C, r], [np.zeros((1, 3)), 1]])
        return T

    @staticmethod
    @jit
    def to_components(T):
        """
        Decompose an SE(3) matrix into a rotation matrix and translation vector.
        """
        C = T[0:3, 0:3]
        r = T[0:3, 3]
        return C, r

    @staticmethod
    @jit
    def random():
        phi = random.uniform(key, (3,), minval=0, maxval=2 * np.pi)
        r = random.uniform(key, (3,), minval=3, maxval=3)
        C = SO3.Exp(phi)
        return SE3.from_components(C, r)

    @staticmethod
    @jit
    def wedge(xi):
        xi = np.array(xi).ravel()
        phi = xi[0:3]
        xi_r = xi[3:]
        Xi_phi = SO3.wedge(phi)
        Xi = np.block([[Xi_phi, xi_r.reshape((-1, 1))], [np.zeros((1, 4))]])
        return Xi

    @staticmethod
    @jit
    def vee(Xi):
        Xi_phi = Xi[0:3, 0:3]
        xi_r = Xi[0:3, 3]
        phi = SO3.vee(Xi_phi)
        return np.vstack((phi, xi_r.reshape((-1, 1))))

    @staticmethod
    @jit
    def exp(Xi):
        Xi_phi = Xi[0:3, 0:3]
        phi = SO3.vee(Xi_phi)
        xi_r = Xi[0:3, 3]
        C = SO3.exp(Xi_phi)
        r = np.dot(SO3.left_jacobian(phi), xi_r.reshape((-1, 1)))
        return SE3.from_components(C, r)

    @staticmethod
    @jit
    def Exp(x):
        phi =x[0:3]
        xi_r = x[3:]
        C = SO3.Exp(phi)
        r = np.dot(SO3.left_jacobian(phi), xi_r.reshape((-1, 1)))
        return SE3.from_components(C, r)

    @staticmethod
    @jit
    def log(T):
        Xi_phi = SO3.log(T[0:3, 0:3])
        r = T[0:3, 3]
        xi_r = np.dot(
            SO3.left_jacobian_inv(SO3.vee(Xi_phi)), r.reshape((-1, 1))
        )
        Xi = np.block([[Xi_phi, xi_r], [np.zeros((1, 4))]])
        return Xi

    @staticmethod
    @jit
    def Log(T):
        phi = SO3.Log(T[0:3, 0:3])
        r = T[0:3, 3]
        xi_r = np.dot(
            SO3.left_jacobian_inv(phi), r.reshape((-1, 1))
        )
        return np.vstack((phi, xi_r.reshape((-1, 1))))

    @staticmethod
    @jit
    def inverse(T):
        C, r = SE3.to_components(T)
        C_inv = C.transpose()
        r_inv = -np.dot(C_inv, r)
        return SE3.from_components(C_inv, r_inv)

    @staticmethod
    @jit
    def odot(b):
        b = np.array(b).ravel()
        X = np.block([[SO3.odot(b[0:3]), b[3:]*np.identity(3)], [np.zeros((1, 6))]])
        return X

    @staticmethod
    @jit
    def adjoint(T):
        C = T[0:3, 0:3]
        r = T[0:3, 3]
        Ad = np.block([[C, np.zeros((3,3))],[np.dot(SO3.wedge(r), C), C]])
        return Ad

    @staticmethod
    @jit
    def _left_jacobian_Q_matrix(phi, rho):
        phi = np.array(phi).ravel()
        rho = np.array(rho).ravel()

        rx = SO3.wedge(rho)
        px = SO3.wedge(phi)

        ph = fast_vector_norm(phi)

        ph2 = ph * ph
        ph3 = ph2 * ph
        ph4 = ph3 * ph
        ph5 = ph4 * ph

        cph = np.cos(ph)
        sph = np.sin(ph)

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
    @jit
    def left_jacobian(xi):

        xi = np.array(xi).ravel()

        phi = xi[0:3]  # rotation part
        rho = xi[3:6]  # translation part

        return lax.cond(
            np.linalg.norm(phi) < SE3._small_angle_tol,
            lambda _: np.identity(6),
            lambda _: SE3._left_jacobian_large_angle(phi, rho),
            None
        )

    @staticmethod
    @jit
    def _left_jacobian_large_angle(phi, rho):
        Q = SE3._left_jacobian_Q_matrix(phi, rho)

        J = SO3.left_jacobian(phi)
        out = np.block([[J, np.zeros((3,3))],[Q, J]])
        return out

    @staticmethod
    @jit
    def left_jacobian_inv(xi):
        xi = np.array(xi).ravel()

        phi = xi[0:3]  # rotation part
        rho = xi[3:6]  # translation part

        return lax.cond(
            np.linalg.norm(phi) < SE3._small_angle_tol,
            lambda _: np.identity(6),
            lambda _: SE3._left_jacobian_inv_large_angle(phi, rho),
        None)

    @staticmethod
    @jit
    def _left_jacobian_inv_large_angle(phi, rho):
        Q = SE3._left_jacobian_Q_matrix(phi, rho)

        J_inv = SO3.left_jacobian_inv(phi)
        out = np.block([[J_inv, np.zeros((3,3))],[-np.dot(J_inv, np.dot(Q, J_inv)), J_inv]])
        return out

    @staticmethod
    @jit
    def right_jacobian(xi):
        return SE3.left_jacobian(-xi)

    @staticmethod
    @jit
    def right_jacobian_inv(xi):
        return SE3.left_jacobian_inv(-xi)
