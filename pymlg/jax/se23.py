from .base import MatrixLieGroup
import jax.numpy as jnp
import numpy as onp
from jax import jit, lax
from .so3 import SO3
from .se3 import SE3


class SE23(MatrixLieGroup):
    """
    An instantiation-free implementation of the SE_2(3) matrix Lie group.
    """

    dof = 9
    matrix_size = 5
    _identity = None
    
    @staticmethod
    @jit
    def random():
        return super(SE23, SE23).random()
    
    @staticmethod
    @jit
    def from_components(C, v, r):
        """
        Construct an :math:`SE_2(3)` matrix from attitude, velocity, position
        components.

        Parameters
        ----------
        C : ndarray with shape (3,3)
            DCM/rotation matrix
        v : list[float] or ndarray with size 3
            velocity vector
        r : list[float] or ndarray with size 3
            position vector

        Returns
        -------
        ndarray with shape (5,5)
            :math:`SE_2(3)` matrix
        """
        X = jnp.zeros((5, 5))
        X = X.at[:3,:3].set(C)
        X = X.at[:3,3].set(v.ravel())
        X = X.at[:3,4].set(r.ravel())
        X = X.at[3,3].set(1)
        X = X.at[4,4].set(1)
        # X = jnp.block([[C, v, r], [SE23._zeros_2_3, SE23._eye_2]])
        return X

    @staticmethod
    @jit
    def to_components(X):
        """
        Extract rotation, velocity, position from SE_2(3) matrix.
        """
        C = X[0:3, 0:3]
        v = X[0:3, 3].reshape((-1, 1))
        r = X[0:3, 4].reshape((-1, 1))

        return (C, v, r)

    @staticmethod
    @jit
    def wedge(xi):
        xi = jnp.array(xi).ravel()
        xi_phi = xi[0:3]
        xi_v = xi[3:6].reshape((-1, 1))
        xi_r = xi[6:9].reshape((-1, 1))

        # Xi = jnp.block([[SO3.wedge(xi_phi), xi_v, xi_r], [onp.zeros((2, 5))]])
        Xi = jnp.zeros((5, 5))
        Xi = Xi.at[0:3, 0:3].set(SO3.wedge(xi_phi))
        Xi = Xi.at[0:3, 3].set(xi_v.ravel())
        Xi = Xi.at[0:3, 4].set(xi_r.ravel())

        return Xi

    @staticmethod
    @jit
    def vee(Xi):

        # xi = jnp.vstack(
        #     (
        #         SO3.vee(Xi[0:3, 0:3]),
        #         Xi[0:3, 3].reshape((-1, 1)),
        #         Xi[0:3, 4].reshape((-1, 1)),
        #     )
        # )
        xi = jnp.zeros((9, 1))
        xi = xi.at[0:3].set(SO3.vee(Xi[0:3, 0:3])) 
        xi = xi.at[3:6].set(Xi[0:3, 3].reshape((-1, 1)))
        xi = xi.at[6:9].set(Xi[0:3, 4].reshape((-1, 1)))
        return xi

    @staticmethod
    @jit
    def exp(Xi):
        Xi_phi = Xi[0:3, 0:3]
        xi_phi = SO3.vee(Xi_phi)
        xi_v = Xi[0:3, 3].reshape((-1, 1))
        xi_r = Xi[0:3, 4].reshape((-1, 1))
        C = SO3.exp(Xi_phi)

        J_left = SO3.left_jacobian(xi_phi)
        v = jnp.dot(J_left, xi_v)
        r = jnp.dot(J_left, xi_r)
        return SE23.from_components(C, v, r)

    @staticmethod
    @jit
    def log(X):
        (C, v, r) = SE23.to_components(X)
        phi = SO3.Log(C)
        J_left_inv = SO3.left_jacobian_inv(phi)

        Xi = jnp.zeros((5, 5))
        Xi = Xi.at[0:3, 0:3].set(SO3.wedge(phi))
        Xi = Xi.at[0:3, 3].set(jnp.dot(J_left_inv, v).ravel())
        Xi = Xi.at[0:3, 4].set(jnp.dot(J_left_inv, r).ravel())
        # Xi = jnp.block([
        #     [SO3.wedge(phi), jnp.dot(J_left_inv, v), jnp.dot(J_left_inv, r)],
        #     [onp.zeros((2, 5))]
        # ])

        return Xi

    @staticmethod
    @jit
    def odot(b):
        b = jnp.array(b).ravel()
        # M1 = jnp.vstack([SE3.odot(b[0:4]), onp.zeros((1,6))])
        # M2 = jnp.vstack([b[4]*onp.identity(3), onp.zeros((2, 3))])
        # X = jnp.hstack([M1, M2])
        X = jnp.zeros((5, 9))
        X = X.at[0:4, 0:6].set(SE3.odot(b[0:4]))
        X = X.at[0:3, 6:9].set(b[4] * jnp.identity(3))
        return X
    
    @staticmethod
    @jit
    def Exp(x):
        xi_phi = x[0:3]
        xi_v = x[3:6].reshape((-1, 1))
        xi_r = x[6:9].reshape((-1, 1))
        C = SO3.Exp(xi_phi)

        J_left = SO3.left_jacobian(xi_phi)
        v = jnp.dot(J_left, xi_v)
        r = jnp.dot(J_left, xi_r)
        return SE23.from_components(C, v, r)
    
    @staticmethod
    @jit
    def Log(X):
        return SE23.vee(SE23.log(X))

    @staticmethod
    @jit
    def inverse(X):
        C = X[:3, :3]
        v = X[:3, 3]
        r = X[:3, 4] 
        C_inv = C.T
        v_inv = -C_inv @ v
        r_inv = -C_inv @ r

        X_inv = SE23.from_components(C_inv, v_inv, r_inv)
        return X_inv

    @staticmethod
    @jit
    def adjoint(X):
        C, v, r = SE23.to_components(X)
        # Ad = jnp.block([
        #     [C, jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        #     [SO3.wedge(v).dot(C), C, jnp.zeros((3, 3))],
        #     [SO3.wedge(r).dot(C), jnp.zeros((3, 3)), C]
        # ])
        # Ad = jnp.zeros((9, 9))
        # Ad[0:3, 0:3] = C
        # Ad[3:6, 0:3] = SO3.wedge(v).dot(C)
        # Ad[3:6, 3:6] = C
        # Ad[6:9, 0:3] = SO3.wedge(r).dot(C)
        # Ad[6:9, 6:9] = C

        Ad = jnp.zeros((9, 9))
        Ad = Ad.at[0:3, 0:3].set(C)
        Ad = Ad.at[3:6, 0:3].set(SO3.wedge(v).dot(C))
        Ad = Ad.at[3:6, 3:6].set(C)
        Ad = Ad.at[6:9, 0:3].set(SO3.wedge(r).dot(C))
        Ad = Ad.at[6:9, 6:9].set(C)
        return Ad

    @staticmethod
    @jit
    def left_jacobian(xi):
        xi = jnp.array(xi).ravel()
        xi_phi = xi[0:3]
        xi_v = xi[3:6].reshape((-1, 1))
        xi_r = xi[6:9].reshape((-1, 1))

        return lax.cond(
            jnp.linalg.norm(xi_phi) < SE23._small_angle_tol,
            lambda _: onp.identity(9),
            lambda _: SE23._left_jacobian_large_angle(xi_phi, xi_v, xi_r),
            None
        )

    @staticmethod
    @jit
    def _left_jacobian_large_angle(xi_phi, xi_v, xi_r):
        
        Q_v = SE3._left_jacobian_Q_matrix(xi_phi, xi_v)
        Q_r = SE3._left_jacobian_Q_matrix(xi_phi, xi_r)
        J = SO3.left_jacobian(xi_phi)
        # J_left = jnp.block([
        #     [J, jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        #     [Q_v, J, jnp.zeros((3, 3))],
        #     [Q_r, jnp.zeros((3, 3)), J]
        # ])
        J_left = jnp.zeros((9, 9))
        J_left = J_left.at[0:3, 0:3].set(J)
        J_left = J_left.at[3:6, 0:3].set(Q_v)
        J_left = J_left.at[3:6, 3:6].set(J)
        J_left = J_left.at[6:9, 0:3].set(Q_r)
        J_left = J_left.at[6:9, 6:9].set(J)

        return J_left

    @staticmethod
    @jit
    def left_jacobian_inv(xi):
        xi = jnp.array(xi).ravel()
        xi_phi = xi[0:3]
        xi_v = xi[3:6].reshape((-1, 1))
        xi_r = xi[6:9].reshape((-1, 1))

        return lax.cond(
            jnp.linalg.norm(xi_phi) < SE23._small_angle_tol,
            lambda _: onp.identity(9),
            lambda _: SE23._left_jacobian_inv_large_angle(xi_phi, xi_v, xi_r),
            None
        )

    @staticmethod
    @jit
    def _left_jacobian_inv_large_angle(xi_phi, xi_v, xi_r):

        Q_v = SE3._left_jacobian_Q_matrix(xi_phi, xi_v)
        Q_r = SE3._left_jacobian_Q_matrix(xi_phi, xi_r)
        J_inv = SO3.left_jacobian_inv(xi_phi)

        # J_left_inv = jnp.block([
        #     [J_inv, jnp.zeros((3, 3)), jnp.zeros((3, 3))],
        #     [-J_inv.dot(Q_v).dot(J_inv), J_inv, jnp.zeros((3, 3))],
        #     [-J_inv.dot(Q_r).dot(J_inv), jnp.zeros((3, 3)), J_inv] 
        # ])

        J_left_inv = jnp.zeros((9, 9))
        J_left_inv = J_left_inv.at[0:3, 0:3].set(J_inv)
        J_left_inv = J_left_inv.at[3:6, 0:3].set(-J_inv.dot(Q_v).dot(J_inv))
        J_left_inv = J_left_inv.at[3:6, 3:6].set(J_inv)
        J_left_inv = J_left_inv.at[6:9, 0:3].set(-J_inv.dot(Q_r).dot(J_inv))
        J_left_inv = J_left_inv.at[6:9, 6:9].set(J_inv)

        return J_left_inv
