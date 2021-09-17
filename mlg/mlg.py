from scipy.linalg import expm, logm
import numpy as np

class MatrixLieGroup:

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
