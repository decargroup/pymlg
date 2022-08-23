from scipy.linalg import expm, logm
import numpy as np


class MatrixLieGroup:
    """
    Base class inherited by all groups, providing a few group-general
    methods.

    """

    _small_angle_tol = 1e-7
    dof = None

    @staticmethod
    def random():
        """
        Return a random element of the group.
        """
        raise NotImplementedError()

    @staticmethod
    def wedge(x):
        """
        Wedge operator :math:`(\cdot)^\\wedge: \mathbb{R}^n \\to \mathfrak{g}`. 
        """
        raise NotImplementedError()

    @staticmethod
    def vee(Xi):
        """
        Vee operator :math:`(\cdot)^\\vee: \mathfrak{g} \\to  \mathbb{R}^n`. 
        """
        raise NotImplementedError()

    @staticmethod
    def exp(Xi):
        """
        Exponential map from algebra to group.

        .. math::
            \mathrm{exp}: \mathfrak{g} \\to G
        """
        return expm(Xi)

    @staticmethod
    def log(X):
        """
        Logarithmic map from algebra to group.

        .. math::
            \mathrm{log}: G \\to \mathfrak{g}
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

        .. math::
            \mathbf{a}^\wedge \mathbf{b} = \mathbf{b}^\odot \mathbf{a}
        """
        raise NotImplementedError()

    @staticmethod
    def adjoint(X):
        """
        Adjoint representation of *group* element.

        .. math::
            \mathrm{Ad}(\mathbf{X})
        """
        raise NotImplementedError()

    @staticmethod
    def adjoint_algebra(Xi):
        """
        Adjoint representation of *algebra* element.

        .. math::
            \mathrm{ad}(\mathbf{\Xi})
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
        Group right jacobian evaluated at x in R^n. Requires the subclass to
        implement `left_jacobian()`.
        """
        return cls.left_jacobian(-x)

    @classmethod
    def right_jacobian_inv(cls, x):
        """
        Inverse of group right jacobian evaluated at x in R^n. Requires the
        subclass to implement `left_jacobian()`.
        """
        return np.linalg.inv(cls.right_jacobian(x))

    @classmethod
    def Exp(cls, x):
        """
        Shortcut method: R^n to group directly.


        .. math::
            \mathrm{Exp}: \mathbb{R}^n \\to G

        """
        return cls.exp(cls.wedge(x))

    @classmethod
    def Log(cls, X):
        """
        Shortcut method: group to R^n directly.

        .. math::
            \mathrm{Log}: G \\to \mathbb{R}^n

        """
        return cls.vee(cls.log(X))

    @classmethod
    def identity(cls):
        """
        Returns an identity matrix of the group.
        """
        return np.identity(cls.dof)
