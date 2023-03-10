from pylie.numpy import MatrixLieGroup
from ._impl import SO3 as SO3cpp
from typing import Type

class SO3(MatrixLieGroup):
    dof = SO3cpp.dof
    matrix_size = SO3cpp.matrix_size
    _small_angle_tol = SO3cpp.small_angle_tol
    @staticmethod
    def Exp(x):
        return SO3cpp.Exp(x)
    
    @staticmethod
    def Log(x):
        return SO3cpp.Log(x).reshape((SO3cpp.dof,1))
    
    @staticmethod
    def exp(x):
        return SO3cpp.exp(x)
    
    @staticmethod
    def log(x):
        return SO3cpp.log(x)
    
    @staticmethod
    def adjoint(x):
        return SO3cpp.adjoint(x)
    
    @staticmethod
    def left_jacobian(x):
        return SO3cpp.left_jacobian(x)
    
    @staticmethod
    def left_jacobian_inv(x):
        return SO3cpp.left_jacobian_inv(x)
    
    @staticmethod
    def odot(x):
        return SO3cpp.odot(x)
    
    @staticmethod
    def wedge(x):
        return SO3cpp.wedge(x)
    
    @staticmethod
    def random():
        return SO3cpp.random()
    
    @staticmethod
    def identity():
        return SO3cpp.identity()
    
    @staticmethod
    def vee(x):
        return SO3cpp.vee(x).reshape((SO3cpp.dof,1))
    
    @staticmethod
    def inverse(x):
        return SO3cpp.inverse(x)


