# Pylie
An instantiation-free python library for common matrix Lie group operations. Functions in this repo operate entirely on numpy arrays, with the intention of minimizing overhead and keeping the implementation as simple as possible.

To install, go to this directory and run

    pip install -e .

## Example 

```python
from pylie import SE3 

# Random pose
T = SE3.random()

# R^n to group directly (using "Capital" notation)
x = np.array([0.1, 0.2, 0.3, 4, 5, 6])
T = SE3.Exp(x)

# Group to R^n directly
x = SE3.Log(T)

# Wedge, vee
Xi = SE3.wedge(x)
x = SE3.vee(x)

# Actual exp/log maps 
T = SE3.exp(Xi)
Xi = SE3.log(T)

# Adjoint representation of group element
A = SE3.adjoint(T)

# Group left/right jacobians
J_L = SE3.left_jacobian(x)
J_R = SE3.right_jacobian(x)
J_L_inv = SE3.left_jacobian_inv(x)
J_R_inv = SE3.right_jacobian_inv(x)

```

## Running Tests
If you use VS Code, you should be able to enable the VS Code testing feature using pytest. Otherwise, you can run tests from the command line when inside this folder using

    pytest tests