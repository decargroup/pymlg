from .base import MatrixLieGroup
import torch
from .utils import *

class SE23(MatrixLieGroup):
    """
    An torch instantiation-free implementation of the SE_2(3) matrix Lie group.

    ALL functions are made to accept a batch: [N x ...].
    """

    dof = 9

    @staticmethod
    def from_components(rot : torch.Tensor, v : torch.Tensor, r : torch.Tensor):
        """
        Construct a batch of :math:`SE_2(3)` matricies from attitude, velocity, position 
        components.

        Parameters
        ----------
        rot : torch.Tensor with shape (N, 3, 3) or (N, 3)
            batch of rotations, either rotation matricies or rotation vectors
        v : torch.Tensor with size (N, 3)
            batch of velocity vectors
        r : torch.Tensor with size (N, 3)
            batch of position vectors

        Returns
        -------
        ndarray with shape (N, 5, 5)
            :math:`SE_2(3)` matrix
        """
        
        # firstly, determine if the given batch of rotation is a set of valid rotation matricies
        pass