# Pylie
An instantiation-free python library for common matrix Lie group operations. Functions in this repo operate entirely on numpy arrays, with the intention of minimizing overhead and keeping the implementation as simple as possible.


__Definitely a work in progress! :)__

## Installation
Begin by cloning this repo somewhere. To install, go to the clone directory and run

    pip install -e .

## Example 

```python
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

```

**Note:** functions which output "vectors", such as `SE2.Log(T)` all return a 2D numpy array with dimensions `(n, 1)`.


## Running Tests
If you use VS Code, you should be able to enable the VS Code testing feature using pytest. Otherwise, you can run tests from the command line when inside this folder using

    pytest tests

## Credit to UTIAS's STARS group
We would also like to advertise the [UTIAS STARS Lie group library](https://github.com/utiasSTARS/liegroups), which has very similar functionality and inspired some of the specific function implementations in this repo (such as the SE2 Jacobian functions). It is also definitely a more "mature" work in the sense of better documentation, and probably less bugs. 

We wanted to make our own mainly for abiding by our group's conventions, as well as to enjoy the simplicity of an instantiation-free implementation. That is, all functions in this repo are static methods operating on numpy arrays. 
