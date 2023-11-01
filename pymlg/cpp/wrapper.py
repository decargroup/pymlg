from pymlg.numpy import MatrixLieGroup
from pymlg.numpy import SO3 as SO3np
from pymlg.numpy import SE3 as SE3np
from pymlg.numpy import SE23 as SE23np
from pymlg.numpy import SL3 as SL3np
from ._impl import SO3 as SO3cpp
from ._impl import SE3 as SE3cpp
from ._impl import SE23 as SE23cpp
from ._impl import SL3 as SL3cpp
from typing import Type

class SO3(SO3np):
    """
    Special Orthogonal Group in 3D.
    """
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
    def adjoint_algebra(x):
        return SO3cpp.adjoint_algebra(x)

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
    """ 
    Special Euclidean Group in 3D.
    """

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
    def adjoint_algebra(x):
        return SE3cpp.adjoint_algebra(x)

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

class SE23(SE23np):
    """
    "Extended" Special Euclidean Group in 3D.
    """
    dof = SE23cpp.dof
    matrix_size = SE23cpp.matrix_size
    _small_angle_tol = SE23cpp.small_angle_tol
    @staticmethod
    def Exp(x):
        return SE23cpp.Exp(x)
    
    @staticmethod
    def Log(x):
        return SE23cpp.Log(x).reshape((SE23cpp.dof,1))
    
    @staticmethod
    def exp(x):
        return SE23cpp.exp(x)
    
    @staticmethod
    def log(x):
        return SE23cpp.log(x)
    
    @staticmethod
    def adjoint(x):
        return SE23cpp.adjoint(x)
    
    @staticmethod
    def adjoint_algebra(x):
        return SE23cpp.adjoint_algebra(x)

    @staticmethod
    def left_jacobian(x):
        return SE23cpp.left_jacobian(x)
    
    @staticmethod
    def left_jacobian_inv(x):
        return SE23cpp.left_jacobian_inv(x)
    
    @staticmethod
    def odot(x):
        return SE23cpp.odot(x)
    
    @staticmethod
    def wedge(x):
        return SE23cpp.wedge(x)
    
    @staticmethod
    def random():
        return SE23cpp.random()
    
    @staticmethod
    def identity():
        return SE23cpp.identity()
    
    @staticmethod
    def vee(x):
        return SE23cpp.vee(x).reshape((SE23cpp.dof,1))
    
    @staticmethod
    def inverse(x):
        return SE23cpp.inverse(x)
    
class SL3(SL3np):
    """ 
    Special Linear Group in 3D.
    """
    dof = SL3cpp.dof
    matrix_size = SL3cpp.matrix_size
    _small_angle_tol = SL3cpp.small_angle_tol
    @staticmethod
    def Exp(x):
        return SL3cpp.Exp(x)
    
    @staticmethod
    def Log(x):
        return SL3cpp.Log(x).reshape((SL3cpp.dof,1))
    
    @staticmethod
    def exp(x):
        return SL3cpp.exp(x)
    
    @staticmethod
    def log(x):
        return SL3cpp.log(x)
    
    @staticmethod
    def adjoint(x):
        return SL3cpp.adjoint(x)
    
    @staticmethod
    def adjoint_algebra(x):
        return SL3cpp.adjoint_algebra(x)

    @staticmethod
    def left_jacobian(x):
        return SL3cpp.left_jacobian(x)
    
    @staticmethod
    def left_jacobian_inv(x):
        return SL3cpp.left_jacobian_inv(x)
    
    @staticmethod
    def odot(x):
        return SL3cpp.odot(x)
    
    @staticmethod
    def wedge(x):
        return SL3cpp.wedge(x)
    
    @staticmethod
    def random():
        return SL3cpp.random()
    
    @staticmethod
    def identity():
        return SL3cpp.identity()
    
    @staticmethod
    def vee(x):
        return SL3cpp.vee(x).reshape((SL3cpp.dof,1))
    
    @staticmethod
    def inverse(x):
        return SL3cpp.inverse(x)
