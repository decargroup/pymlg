.. pylie documentation master file, created by
   sphinx-quickstart on Mon Aug 22 21:49:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pylie
=================================

.. toctree::
   :maxdepth: 3
   :caption: Contents:


An instantiation-free python library for common matrix Lie group operations. Functions in this repo operate entirely on numpy arrays, with the intention of minimizing overhead and keeping the implementation as simple as possible.

Example
-------

.. code-block:: python 

    from pylie import SE2 
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
    
API
==================

.. autoclass:: pylie.SO2
    :members:
    :inherited-members: Exp, Log, right_jacobian, right_jacobian_inv, identity
    :undoc-members:

.. autoclass:: pylie.SE2
    :members: 
    :inherited-members: Exp, Log, right_jacobian, right_jacobian_inv, identity
    :undoc-members:

.. autoclass:: pylie.SO3
    :members:
    :inherited-members: Exp, Log, right_jacobian, right_jacobian_inv, identity
    :undoc-members:

.. autoclass:: pylie.SE3
    :members: 
    :inherited-members: Exp, Log, right_jacobian, right_jacobian_inv, identity
    :undoc-members:
    

.. autoclass:: pylie.SE23
    :members: 
    :inherited-members: Exp, Log, right_jacobian, right_jacobian_inv, identity
    :undoc-members:
    