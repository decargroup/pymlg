from .base import MatrixLieGroup
import jax.numpy as jnp
from jax import jit

class SO2(MatrixLieGroup):
    """
    An instantiation-free implementation of the SO2 matrix Lie group.
    """

    dof = 1
    matrix_size = 2

    @staticmethod
    @jit
    def wedge(phi):
        phi = phi.reshape((1,))
        X = jnp.zeros_like(phi, shape=(2, 2))
        X = X.at[0,1].set(-phi[0])
        X = X.at[1,0].set(phi[0])
        return X

    @staticmethod
    @jit
    def vee(X):
        phi = X[1, 0]
        return phi

    @staticmethod
    @jit
    def exp(Xi):
        phi = SO2.vee(Xi)
        X = jnp.array([[jnp.cos(phi), -jnp.sin(phi)], [jnp.sin(phi), jnp.cos(phi)]])
        return X

    @staticmethod
    @jit
    def log(X):
        phi = jnp.arctan2(X[1, 0], X[0, 0])
        return SO2.wedge(phi)
    
    @staticmethod
    @jit
    def Exp(phi):
        phi = phi.reshape((1,))[0]
        X = jnp.array([[jnp.cos(phi), -jnp.sin(phi)], [jnp.sin(phi), jnp.cos(phi)]])
        return X

    @staticmethod
    @jit
    def Log(X):
        return jnp.arctan2(X[1, 0], X[0, 0])


    @staticmethod
    @jit
    def left_jacobian(x):
        return jnp.array([[1]])

    @staticmethod
    @jit
    def left_jacobian_inv(x):
        return jnp.array([[1]])

    @staticmethod
    @jit
    def right_jacobian(x):
        return jnp.array([[1]])

    @staticmethod
    @jit
    def right_jacobian_inv(x):
        return jnp.array([[1]])

    @staticmethod
    @jit
    def adjoint(C):
        return jnp.array([[1]])

    @staticmethod
    @jit
    def odot(b):
        b = jnp.array(b).ravel()
        return jnp.array([-b[1], b[0]]).reshape((-1, 1))