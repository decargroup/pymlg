from .base import MatrixLieGroup
import numpy as np
import math

class SL3(MatrixLieGroup):
    """
    An instantiation-free implementation of the SO3 matrix Lie group.
    """

    dof = 8

    @staticmethod
    def random():
        v = np.random.randn(8, 1)
        return SL3.Exp(v)

    @staticmethod 
    def wedge(xi):
        xi = xi.ravel()
        X = np.array(
            [
                [xi[3]+xi[4], -xi[2]+xi[5], xi[0]],
                [xi[2]+xi[5], xi[3]-xi[4], xi[1]],
                [xi[6], xi[7], -2*xi[3]]
            ]
        )
        return X

    @staticmethod
    def cross(xi):
        return SL3.wedge(xi)

    @staticmethod
    def vee(X):
        xi = np.array([[X[0, 2]], 
                        [X[1, 2]], 
                        [1/2*(X[1, 0]-X[0,1])],
                        [-1/2*X[2,2]],
                        [1/2*(X[0,0]-X[1,1])],
                        [1/2*(X[1,0]+X[0,1])],
                        [X[2,0]],
                        [X[2,1]]
                        ])
        return xi

    @staticmethod
    def left_jacobian(xi):
        """Computes the Left Jacobian of SL(3) numerically.

        """
        X = SL3.Exp(xi)
        exp_inv = SL3.inverse(X)
        J_fd = np.zeros((SL3.dof, SL3.dof))
        h = 1e-8
        for i in range(SL3.dof):
            dx = np.zeros(SL3.dof)
            dx[i] =  h 
            J_fd[:, i] = (SL3.Log(SL3.Exp(xi + dx) @ exp_inv) / h).ravel()
            #J_fd[:, i] = np.imag(SL3.Exp (dx)@ X) / h
        return J_fd


    @staticmethod
    def odot(p):
        """"
        This expression helps obtain p^\odot where
        Xi * p = p_odot * xi,
        and p is a vector of 3x1, usually representing position.
        """
        p = np.array(p).ravel()
        p_odot = np.array([[p[2], 0, -p[1], p[0], p[0], p[1], 0, 0],
                            [0, p[2], p[0], p[1], -p[1], p[0], 0, 0],
                            [0, 0, 0, -2*p[2], 0, 0, p[0], p[1]]])
        return p_odot

    @staticmethod
    def adjoint(H):

        """" Adjoint representation of GROUP element.
        Obtained from Section 7.4 of Lie Groups for Computer Vision by Eade"""
        alg = np.array([[0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, -1, 0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, -1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, -2, 0, 0, 0, 0]])
        alg_inv = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, -1/2, 0, 1/2, 0, 0, 0, 0, 0],
                            [1/2, 0, 0, 0, 1/2, 0, 0, 0, 0],
                            [1/2, 0, 0, 0, -1/2, 0, 0, 0, 0],
                            [0, 1/2, 0, 1/2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0]])
        H_inv_T = SL3.inverse(H).T
        C_H = np.block([[H[0,0]*H_inv_T, H[0,1]*H_inv_T, H[0,2]*H_inv_T],
                        [H[1,0]*H_inv_T, H[1,1]*H_inv_T, H[1,2]*H_inv_T],
                        [H[2,0]*H_inv_T, H[2,1]*H_inv_T, H[2,2]*H_inv_T]])

        Adj = alg_inv @ C_H @ alg
        return Adj

    @staticmethod
    def adjoint_algebra(xi):
        """
        Adjoint representation of ALGEBRA element.
        [Xi_1, Xi_2]^\vee = ad(Xi_1)xi_2
        """
        xi = xi.ravel()
        
        ad = np.array([
                        [3*xi[3]+xi[4], -(xi[2]-xi[5]), xi[1], -3*xi[0], -xi[0], -xi[1], 0, 0],
                        [xi[2]+xi[5], 3*xi[3]-xi[4], -xi[0], -3*xi[1], xi[1], -xi[0], 0, 0],
                        [xi[7]/2, -xi[6]/2, 0, 0, 2*xi[5], -2*xi[4], xi[1]/2, -xi[0]/2],
                        [-xi[6]/2, -xi[7]/2, 0, 0, 0, 0, xi[0]/2, xi[1]/2],
                        [-xi[6]/2, xi[7]/2, 2*xi[5], 0, 0, -2*xi[2], xi[0]/2, -xi[1]/2],
                        [-xi[7]/2, -xi[6]/2, -2*xi[4], 0, 2*xi[2], 0, xi[1]/2, xi[0]/2],
                        [0, 0, xi[7], 3*xi[6], xi[6], xi[7], -3*xi[3]-xi[4], -(xi[2]+xi[5])],
                        [0, 0, -xi[6], 3*xi[7], -xi[7], xi[6], xi[2]-xi[5], -(3*xi[3]-xi[4])]
                        ])
        return ad
    
    """ @staticmethod
    def exp(Xi, ord = 5):


        res = np.eye(3)
        for i in range(1,ord):
            a = math.factorial(i)
            res += 1/a * np.linalg.matrix_power(Xi, i)
        return res """