.. pymlg documentation master file, created by
   sphinx-quickstart on Mon Aug 22 21:49:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pymlg
=================================

An instantiation-free python library for common matrix Lie group operations. Functions in this repo operate entirely on numpy arrays, with the intention of minimizing overhead and keeping the implementation as simple as possible.

**All class methods are static.** This allows for an easy, simple, consistent, instantiation-free use of the library. 

.. toctree::
    :maxdepth: 4
    :caption: Contents:

    Introduction <self>

Example
-------

.. code-block:: python 

    from pymlg import SE2 
    import numpy as np

    # Random pose
    T = SE2.random()

    # R^n to group directly (using "Capital" notation)
    x = np.array([0.1, 2, 3])
    T = SE2.Exp(x)

    # Group to R^n directly
    x = SE2.Log(T)

    # Wedge, vee
    Xi = SE2.wedge(x)
    x = SE2.vee(Xi)

    # Actual exp/log maps 
    T = SE2.exp(Xi)
    Xi = SE2.log(T)

    # Adjoint representation of group element
    A = SE2.adjoint(T)

    # Group left/right jacobians
    J_L = SE2.left_jacobian(x)
    J_R = SE2.right_jacobian(x)
    J_L_inv = SE2.left_jacobian_inv(x)
    J_R_inv = SE2.right_jacobian_inv(x)

    # odot operator (defined such that wedge(a) * b = odot(b) * a)
    b = np.array([1, 2, 3]) 
    B = SE2.odot(b)

Full Documentation
------------------
Click on the table entries below to go to each class' documentation.

.. autosummary:: 
    :toctree: _autosummary/
    :recursive:
    :template: class.rst
    :nosignatures:

    
    pymlg.SO2
    pymlg.SO3
    pymlg.SE2
    pymlg.SE3
    pymlg.SE23
    pymlg.SL3
    pymlg.MatrixLieGroup



