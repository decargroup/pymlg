from scipy.linalg import expm, logm
import numpy as np


class MatrixLieGroup:

    _small_angle_tol = 1e-14

    @staticmethod
    def random():
        raise NotImplementedError()

    @staticmethod
    def wedge(x):
        raise NotImplementedError()

    @staticmethod
    def vee(x):
        raise NotImplementedError()

    @staticmethod
    def exp(x):
        """
        Default implementation, can be overridden.
        """
        return expm(x)

    @staticmethod
    def log(x):
        """
        Default implementation, can be overridden.
        """
        return logm(x)

    @staticmethod
    def inverse(X):
        """
        Default implementation, can be overridden.
        """
        return np.linalg.inv(X)

    @staticmethod
    def Adjoint(X):
        raise NotImplementedError()

    @staticmethod
    def adjoint(Xi):
        raise NotImplementedError()

    @staticmethod
    def left_jacobian(x):
        raise NotImplementedError()

    @classmethod
    def left_jacobian_inv(cls, x):
        return np.linalg.inv(cls.left_jacobian(x))

    @classmethod
    def right_jacobian(cls, x):
        return cls.left_jacobian(-x)

    @classmethod
    def right_jacobian(cls, x):
        return np.linalg.inv(cls.right_jacobian(x))

    @classmethod
    def Exp(cls, x):
        """
        Shortcut method.
        """
        return cls.exp(cls.wedge(x))

    @classmethod
    def Log(cls, x):
        """
        Shortcut method.
        """
        return cls.vee(cls.log(x))

    @staticmethod
    def normalize(X):
        raise NotImplementedError()