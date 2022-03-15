from scipy.linalg import expm, logm
import numpy as np


class MatrixLieGroup:

    _small_angle_tol = 1e-7

    @staticmethod
    def synthesize():
        raise NotImplementedError()

    @staticmethod
    def random():
        """
        Return a random element of the group.
        """
        raise NotImplementedError()

    @staticmethod
    def wedge(x):
        """
        Wedge operator from R^n to algebra.
        """
        raise NotImplementedError()

    @staticmethod
    def vee(Xi):
        """
        Vee operator from algebra to R^n.
        """
        raise NotImplementedError()

    @staticmethod
    def exp(Xi):
        """
        Exponential map from algebra to group.
        """
        return expm(Xi)

    @staticmethod
    def log(X):
        """
        Logarithmic map from algebra to group.
        """
        return logm(X)

    @staticmethod
    def inverse(X):
        """
        Group inverse.
        """
        return np.linalg.inv(X)

    @staticmethod
    def normalize(X):
        """
        Eliminates rounding errors: ensures matrix is proper group element.
        """
        raise NotImplementedError()

    @staticmethod
    def odot(b):
        """
        odot operator as defined in Barfoot. I.e., an operator on an element of
        R^n such that

        a^wedge b = b^odot a
        """
        raise NotImplementedError()

    @staticmethod
    def adjoint(Xi):
        """
        Adjoint representation of GROUP element.
        """
        raise NotImplementedError()

    @staticmethod
    def adjoint_algebra(Xi):
        """
        Adjoint representation of ALGEBRA element.
        """
        raise NotImplementedError()

    @staticmethod
    def left_jacobian(x):
        """
        Group left jacobian evaluated at x in R^n
        """
        raise NotImplementedError()

    @classmethod
    def left_jacobian_inv(cls, x):
        """
        Inverse of group left jacobian evaluated at x in R^n
        """
        return np.linalg.inv(cls.left_jacobian(x))

    @classmethod
    def right_jacobian(cls, x):
        """
        Group right jacobian evaluated at x in R^n
        """
        return cls.left_jacobian(-x)

    @classmethod
    def right_jacobian(cls, x):
        """
        Inverse of group right jacobian evaluated at x in R^n
        """
        return np.linalg.inv(cls.right_jacobian(x))

    @classmethod
    def Exp(cls, x):
        """
        Shortcut method: R^n to group directly.
        """
        return cls.exp(cls.wedge(x))

    @classmethod
    def Log(cls, x):
        """
        Shortcut method: group to R^n directly.
        """
        return cls.vee(cls.log(x))
