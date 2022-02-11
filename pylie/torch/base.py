import torch

class MatrixLieGroup:

    _small_angle_tol = 1e-7

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
    def Adjoint(X):
        raise NotImplementedError()

    @staticmethod
    def adjoint(Xi):
        raise NotImplementedError()
