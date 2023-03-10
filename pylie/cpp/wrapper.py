from pylie.numpy import MatrixLieGroup
from pylie.numpy import SO3 as SO3np
from pylie.numpy import SE3 as SE3np
from ._impl import SO3 as SO3cpp
from ._impl import SE3 as SE3cpp
from typing import Type

class SO3(SO3np):
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

class SE3(SE3np):
    dof = SE3cpp.dof
    matrix_size = SE3cpp.matrix_size
    _small_angle_tol = SE3cpp.small_angle_tol
    @staticmethod
    def Exp(x):
        return SE3cpp.Exp(x)
    
    @staticmethod
    def Log(x):
        return SE3cpp.Log(x).reshape((SE3cpp.dof,1))
    
    @staticmethod
    def exp(x):
        return SE3cpp.exp(x)
    
    @staticmethod
    def log(x):
        return SE3cpp.log(x)
    
    @staticmethod
    def adjoint(x):
        return SE3cpp.adjoint(x)
    
    @staticmethod
    def left_jacobian(x):
        return SE3cpp.left_jacobian(x)
    
    @staticmethod
    def left_jacobian_inv(x):
        return SE3cpp.left_jacobian_inv(x)
    
    @staticmethod
    def odot(x):
        return SE3cpp.odot(x)
    
    @staticmethod
    def wedge(x):
        return SE3cpp.wedge(x)
    
    @staticmethod
    def random():
        return SE3cpp.random()
    
    @staticmethod
    def identity():
        return SE3cpp.identity()
    
    @staticmethod
    def vee(x):
        return SE3cpp.vee(x).reshape((SE3cpp.dof,1))
    
    @staticmethod
    def inverse(x):
        return SE3cpp.inverse(x)
