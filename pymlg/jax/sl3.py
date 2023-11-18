from .base import MatrixLieGroup
import jax.numpy as np
from jax.scipy.linalg import expm
from scipy.linalg import logm
from jax import jit, grad


# TODO: unfortunately a jax implementation of logm is not available yet.
# this is needed to make this class properly useful. at the moment,
# any functions involving the log map will be slow, not jit-able, and 
# not differentiable.

class SL3(MatrixLieGroup):
    """
    An instantiation-free implementation of the SL3 matrix Lie group.
    """

    dof = 8
    matrix_size = 3

    @staticmethod
    @jit
    def random():
        return super(SL3, SL3).random()
    
    @staticmethod
    @jit
    def wedge(xi):
        xi = np.array(xi).ravel()
        X = np.array(
            [
                [xi[3] + xi[4], -xi[2] + xi[5], xi[0]],
                [xi[2] + xi[5], xi[3] - xi[4], xi[1]],
                [xi[6], xi[7], -2 * xi[3]],
            ]
        )
        return X

    @staticmethod
    @jit
    def vee(X):
        xi = np.array(
            [
                [X[0, 2]],
                [X[1, 2]],
                [1 / 2 * (X[1, 0] - X[0, 1])],
                [-1 / 2 * X[2, 2]],
                [1 / 2 * (X[0, 0] - X[1, 1])],
                [1 / 2 * (X[1, 0] + X[0, 1])],
                [X[2, 0]],
                [X[2, 1]],
            ]
        )
        return xi

    @staticmethod
    def exp(Xi):
        """
        Computes the exponential map of SL(3) numerically.
        """
        return expm(Xi)
    
    @staticmethod
    def log(X):
        return logm(X)
    
    @staticmethod
    @jit
    def Exp(x):
        """
        Computes the exponential map of SL(3) numerically.
        """
        return SL3.exp(SL3.wedge(x))
    
    @staticmethod
    def Log(X):
        return SL3.vee(SL3.log(X))
    
    # @staticmethod
    # @jit
    # def logm(B):
    #     I = np.eye(B.shape[0])
    #     res = np.zeros_like(B)
    #     ITERATIONS = 1000
    #     for k in range(1, ITERATIONS):
    #         res += pow(-1, k+1) * np.linalg.matrix_power(B-I, k)/k
    #     return res


    @staticmethod
    def left_jacobian(xi):
        """
        Computes the Left Jacobian of SL(3) numerically.
        """
        xi = np.array(xi).reshape((-1,1))
        X = SL3.Exp(xi)
        exp_inv = SL3.inverse(X)
        J_fd = np.zeros((SL3.dof, SL3.dof))
        h = 1e-8
        for i in range(SL3.dof):
            dx = np.zeros((SL3.dof, 1))
            dx = dx.at[i].set(h)
            J_fd = J_fd.at[:, i].set((SL3.Log(np.dot(SL3.Exp(xi + dx), exp_inv)) / h).ravel())
        return J_fd

    @staticmethod
    @jit
    def odot(p):
        """
        This expression helps obtain p^\odot where
        Xi * p = p_odot * xi,
        and p is a vector of 3x1, usually representing position.
        """
        p = np.array(p).ravel()
        p_odot = np.array(
            [
                [p[2], 0, -p[1], p[0], p[0], p[1], 0, 0],
                [0, p[2], p[0], p[1], -p[1], p[0], 0, 0],
                [0, 0, 0, -2 * p[2], 0, 0, p[0], p[1]],
            ]
        )
        return p_odot

    @staticmethod
    @jit
    def adjoint(H):
        """
        Adjoint representation of GROUP element.
        Obtained from Section 7.4 of Lie Groups for Computer Vision by Eade
        """
        alg = np.array(
            [
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, -1, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, -1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, -2, 0, 0, 0, 0],
            ]
        )
        alg_inv = np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, -1 / 2, 0, 1 / 2, 0, 0, 0, 0, 0],
                [1 / 2, 0, 0, 0, 1 / 2, 0, 0, 0, 0],
                [1 / 2, 0, 0, 0, -1 / 2, 0, 0, 0, 0],
                [0, 1 / 2, 0, 1 / 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )
        H_inv_T = SL3.inverse(H).T
        C_H = np.block(
            [
                [H[0, 0] * H_inv_T, H[0, 1] * H_inv_T, H[0, 2] * H_inv_T],
                [H[1, 0] * H_inv_T, H[1, 1] * H_inv_T, H[1, 2] * H_inv_T],
                [H[2, 0] * H_inv_T, H[2, 1] * H_inv_T, H[2, 2] * H_inv_T],
            ]
        )

        Adj = alg_inv @ C_H @ alg
        return Adj

    @staticmethod
    @jit
    def adjoint_algebra(Xi):
        """
        Adjoint representation of ALGEBRA element.
        """
        xi = SL3.vee(Xi).ravel()

        ad = np.array(
            [
                [
                    3 * xi[3] + xi[4],
                    -(xi[2] - xi[5]),
                    xi[1],
                    -3 * xi[0],
                    -xi[0],
                    -xi[1],
                    0,
                    0,
                ],
                [
                    xi[2] + xi[5],
                    3 * xi[3] - xi[4],
                    -xi[0],
                    -3 * xi[1],
                    xi[1],
                    -xi[0],
                    0,
                    0,
                ],
                [
                    xi[7] / 2,
                    -xi[6] / 2,
                    0,
                    0,
                    2 * xi[5],
                    -2 * xi[4],
                    xi[1] / 2,
                    -xi[0] / 2,
                ],
                [-xi[6] / 2, -xi[7] / 2, 0, 0, 0, 0, xi[0] / 2, xi[1] / 2],
                [
                    -xi[6] / 2,
                    xi[7] / 2,
                    2 * xi[5],
                    0,
                    0,
                    -2 * xi[2],
                    xi[0] / 2,
                    -xi[1] / 2,
                ],
                [
                    -xi[7] / 2,
                    -xi[6] / 2,
                    -2 * xi[4],
                    0,
                    2 * xi[2],
                    0,
                    xi[1] / 2,
                    xi[0] / 2,
                ],
                [
                    0,
                    0,
                    xi[7],
                    3 * xi[6],
                    xi[6],
                    xi[7],
                    -3 * xi[3] - xi[4],
                    -(xi[2] + xi[5]),
                ],
                [
                    0,
                    0,
                    -xi[6],
                    3 * xi[7],
                    -xi[7],
                    xi[6],
                    xi[2] - xi[5],
                    -(3 * xi[3] - xi[4]),
                ],
            ]
        )
        return ad