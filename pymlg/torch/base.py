import torch

class MatrixLieGroupTorch:
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
    def random(N=1):
        """
        Returns
        -------
        torch.Tensor
            A random element of the group with shape `(N, n, n)`.

        """
        raise NotImplementedError()

    @staticmethod
    def wedge(x):
        """
        Wedge operator :math:`(\cdot)^\\wedge: \mathbb{R}^n \\to \mathfrak{g}`.

        Parameters
        ----------
        x: torch.Tensor with shape `(N, dof, 1)`

        Returns
        -------
        torch.Tensor
            Element of the Lie algebra with shape `(N, n, n)`.
        """
        raise NotImplementedError()

    @staticmethod
    def vee(x):
        """
        Vee operator :math:`(\cdot)^\\vee: \mathfrak{g} \\to  \mathbb{R}^n`.

        Parameters
        ----------
        Xi : torch.Tensor with shape `(N, n, n)`
            Element of the Lie algebra.
        
        Returns
        -------
        torch.Tensor
            Vector with shape (N, n, 1).
        """
        raise NotImplementedError()

    @staticmethod
    def exp(x):
        """
        Default implementation, can be overridden.
        """
        return torch.matrix_exp(x)

    @staticmethod
    def log(x):
        """
        No easy built-in method for logarithm. Must be implemented.
        """
        raise NotImplementedError()

    @staticmethod
    def inverse(X):
        """
        Default implementation, can be overridden.
        """
        return torch.linalg.inv(X)

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
    def adjoint_algebra(Xi):
        """
        Adjoint representation of *algebra* element.

        .. math::
            \mathrm{ad}(\mathbf{\Xi})

        Parameters
        ----------
        Xi : torch.Tensor with shape `(N, n, n)`
            Element of the Lie algebra.

        Returns
        -------
        torch.Tensor
            The matrix :math:`\mathrm{ad}(\mathbf{\Xi})`.
        """
        raise NotImplementedError()

    @staticmethod
    def adjoint(Xi):
        raise NotImplementedError()
    
    @classmethod
    def identity(cls):
        """
        Returns an identity matrix of the group.

        Returns
        -------
        torch.Tensor
            Identity matrix of the group with shape `(N, n, n)`.
        """
        return torch.eye(cls.matrix_size, cls.matrix_size).unsqueeze(0)
    
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
        return torch.linalg.inv(cls.left_jacobian(x))
    
    @staticmethod
    def odot(b):
        """
        odot operator as defined in Barfoot. I.e., an operator on an element of
        R^n such that

        .. math::
            \mathbf{a}^\wedge \mathbf{b} = \mathbf{b}^\odot \mathbf{a}

        Parameters
        ----------
        x : torch.Tensor with size `(N, dof, 1)`

        Returns
        -------
        torch.Tensor
            The matrix :math:`\mathbf{b}^\odot` with shape (N, dof, 1).
        """
        raise NotImplementedError()
    
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
        x : torch.Tensor with size `(N, dof, 1)`

        Returns
        -------
        torch.Tensor
            The matrix :math:`\mathbf{J}_r(\mathbf{x})` with shape `(N, dof, dof)`.

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
        x : torch.Tensor with size `(N, dof, 1)`

        Returns
        -------
        torch.Tensor
            The matrix :math: \mathbf{J}_r^{-1}(\mathbf{x})` with shape `(N, dof, dof)`.

        """
        return cls.left_jacobian_inv(-x)
