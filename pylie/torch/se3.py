from .base import MatrixLieGroup
import torch
from .utils import *
from .so3 import SO3

# DISCLAIMER: this class is very much so un-tested

class SE3(MatrixLieGroup):
    """
    An torch instantiation-free implementation of the SE3 matrix Lie group.

    ALL functions are made to accept a batch: [N x ...].
    """

    dof = 6
    
    @staticmethod
    def _left_jacobian_Q_matrix(xi_phi, xi_rho):

        rx = SO3.wedge(xi_rho)

        px = SO3.wedge(xi_phi)

        ph = xi_phi.norm(p=2, dim=1)
        ph2 = ph * ph
        ph3 = ph2 * ph
        ph4 = ph3 * ph
        ph5 = ph4 * ph

        cph = ph.cos()
        sph = ph.sin()

        m1 = 0.5
        m2 = (ph - sph) / ph3
        m3 = (0.5 * ph2 + cph - 1.) / ph4
        m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

        m2 = m2.unsqueeze_(dim=1).unsqueeze_(dim=2).expand_as(rx)
        m3 = m3.unsqueeze_(dim=1).unsqueeze_(dim=2).expand_as(rx)
        m4 = m4.unsqueeze_(dim=1).unsqueeze_(dim=2).expand_as(rx)

        t1 = rx
        t2 = px.bmm(rx) + rx.bmm(px) + px.bmm(rx).bmm(px)
        t3 = px.bmm(px).bmm(rx) + rx.bmm(px).bmm(px) - 3. * px.bmm(rx).bmm(px)
        t4 = px.bmm(rx).bmm(px).bmm(px) + px.bmm(px).bmm(rx).bmm(px)

        Q = m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

        return Q.squeeze_()