from scipy.linalg import expm, logm
import numpy as np


class MatrixLieGroup:
    """
    Base class inherited by all groups, providing a few group-general
    methods.

    """

    _small_angle_tol = 1e-7

    #:int: The degrees of freedom of the group.
    dof = None
    
    #:int: Matrix dimension of the group.
    matrix_size = None

    def __init__(self):
        raise RuntimeError(
            """
        This class is not meant to be instantiated! The methods are all static,
        which means you can call them directly with

        Y = <class_name_without_brackets>.<method_name>(X)
        """
        )

    @staticmethod
    def random():
        """
        Returns
        -------
        np.ndarray
            A random element of the group with shape `(n,n)`.

        """
        raise NotImplementedError()

    @staticmethod
    def wedge(x):
        """
        Wedge operator :math:`(\cdot)^\\wedge: \mathbb{R}^n \\to \mathfrak{g}`.

        Parameters
        ----------
        x: np.ndarray or List[float] with size `dof`

        Returns
        -------
        np.ndarray
            Element of the Lie algebra with shape `(n,n)`.
        """
        raise NotImplementedError()

    @staticmethod
    def vee(Xi):
        """
        Vee operator :math:`(\cdot)^\\vee: \mathfrak{g} \\to  \mathbb{R}^n`.

        Parameters
        ----------
        Xi : np.ndarray with shape `(n,n)`
            Element of the Lie algebra.
        
        Returns
        -------
        np.ndarray
            Vector with shape (n,1).
        """
        raise NotImplementedError()

    @staticmethod
    def exp(Xi):
        """
        Exponential map from algebra to group.

        .. math::
            \mathrm{exp}: \mathfrak{g} \\to G

        Parameters
        ----------
        Xi : np.ndarray with shape `(n,n)`
            Element of the Lie algebra.
        
        Returns
        -------
        np.ndarray
            Element of the group with shape `(n,n)`.
        """
        return expm(Xi)

    @staticmethod
    def log(X):
        """
        Logarithmic map from algebra to group.

        .. math::
            \mathrm{log}: G \\to \mathfrak{g}

        Parameters
        ----------
        X : np.ndarray with shape `(n,n)`
            Element of the group.

        Returns
        -------
        np.ndarray
            Element of the Lie algebra with shape `(n,n)`.
        """
        return logm(X)

    @staticmethod
    def inverse(X):
        """
        Group inverse.

        Parameters
        ----------
        X : np.ndarray with shape `(n,n)`
            Element of the group.

        Returns
        -------
        np.ndarray
            Element of the group with shape `(n,n)`.
        """
        return np.linalg.inv(X)

    @staticmethod
    def normalize(X):
        """
        Eliminates rounding errors: ensures matrix is proper group element.

        Parameters
        ----------
        X : np.ndarray with shape `(n,n)`
            Element of the group.

        Returns
        -------
        np.ndarray
            Element of the group with shape `(n,n)`.
        """
        raise NotImplementedError()

    @staticmethod
    def odot(b):
        """
        odot operator as defined in Barfoot. I.e., an operator on an element of
        R^n such that

        .. math::
            \mathbf{a}^\wedge \mathbf{b} = \mathbf{b}^\odot \mathbf{a}

        Parameters
        ----------
        x : np.ndarray or List[float] with size `m`

        Returns
        -------
        np.ndarray
            The matrix :math:`\mathbf{b}^\odot` with shape (n, dof).
        """
        raise NotImplementedError()

    @staticmethod
    def adjoint(X):
        """
        Adjoint representation of *group* element.

        .. math::
            \mathrm{Ad}(\mathbf{X})

        Parameters
        ----------
        X : np.ndarray with shape `(n,n)`
            Element of the group.

        Returns
        -------
        np.ndarray
            The matrix :math:`\mathrm{Ad}(\mathbf{X})` with shape `(dof,dof)`.
        """
        raise NotImplementedError()

    @staticmethod
    def adjoint_algebra(Xi):
        """
        Adjoint representation of *algebra* element.

        .. math::
            \mathrm{ad}(\mathbf{\Xi})

        Parameters
        ----------
        Xi : np.ndarray with shape `(n,n)`
            Element of the Lie algebra.

        Returns
        -------
        np.ndarray
            The matrix :math:`\mathrm{ad}(\mathbf{\Xi})`.
        """
        raise NotImplementedError()

    @staticmethod
    def left_jacobian(x):
        """
        Group left jacobian evaluated at x in R^n

        .. math::
            \mathbf{J}_\ell(\mathbf{x})

        Parameters
        ----------
        x : np.ndarray or List[float] with size `dof`

        Returns
        -------
        np.ndarray
            The matrix :math:`\mathbf{J}_\ell(\mathbf{x})` with shape `(dof,dof)`.
        """
        raise NotImplementedError()

    @classmethod
    def left_jacobian_inv(cls, x):
        """
        Inverse of group left jacobian evaluated at x in R^n

        .. math::
            \mathbf{J}_\ell^{-1}(\mathbf{x})

        Parameters
        ----------
        x : np.ndarray or List[float] with size `dof`

        Returns
        -------
        np.ndarray
            The matrix :math:`\mathbf{J}_\ell^{-1}(\mathbf{x})` with shape `(dof,dof)`.
        """
        return np.linalg.inv(cls.left_jacobian(x))

    @classmethod
    def right_jacobian(cls, x):
        """
        Group right jacobian evaluated at x in R^n. This is calculated from the 
        `left_jacobian()` implementation using the fact that 

        .. math::
            \mathbf{J}_r(\mathbf{x}) = \mathbf{J}_\ell(-\mathbf{x})

        which holds for all matrix Lie groups.

        Parameters
        ----------
        x : np.ndarray or List[float] with size `dof`

        Returns
        -------
        np.ndarray
            The matrix :math:`\mathbf{J}_r(\mathbf{x})` with shape `(dof,dof)`.

        """
        return cls.left_jacobian(-x)

    @classmethod
    def right_jacobian_inv(cls, x):
        """
        Inverse of group right jacobian evaluated at x in R^n. This is calculated 
        from the `left_jacobian_inv()` implementation using the fact that 

        .. math::
            \mathbf{J}_r^{-1}(\mathbf{x}) = \mathbf{J}_\ell^{-1}(-\mathbf{x})

        which holds for all matrix Lie groups.

        Parameters
        ----------
        x : np.ndarray or List[float] with size `dof`

        Returns
        -------
        np.ndarray
            The matrix :math: \mathbf{J}_r^{-1}(\mathbf{x})` with shape `(dof,dof)`.

        """
        return cls.left_jacobian_inv(-x)

    @classmethod
    def Exp(cls, x):
        """
        Shortcut method: R^n to group directly.

        .. math::
            \mathrm{Exp}: \mathbb{R}^n \\to G

        Parameters
        ----------
        x : np.ndarray or List[float] with size `dof`
            Vector of exponential coordinates.

        Returns
        -------
        np.ndarray
            Element of the group with shape `(n,n)`.

        """
        return cls.exp(cls.wedge(x))

    @classmethod
    def Log(cls, X):
        """
        Shortcut method: group to R^n directly.

        .. math::
            \mathrm{Log}: G \\to \mathbb{R}^n

        Parameters
        ----------
        X : np.ndarray with shape `(n,n)`
            Element of the group.

        Returns
        -------
        np.ndarray
            Vector of exponential coordinates with shape `(n,1)`.

        """
        return cls.vee(cls.log(X))

    @classmethod
    def identity(cls):
        """
        Returns an identity matrix of the group.

        Returns
        -------
        np.ndarray
            Identity matrix of the group with shape `(n,n)`.
        """
        return np.identity(cls.matrix_size)

def fast_vector_norm(x):
    return np.sqrt(x.dot(x))