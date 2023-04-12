# Pylie - Lie groups with Numpy, Jax, and C++ implementations!
![test package](https://github.com/decargroup/pylie/actions/workflows/test-package.yml/badge.svg)

An instantiation-free python package for common matrix Lie group operations implemented as __pure static classes__. Using pure static classes keeps the usage extremely simple while still allowing for abstraction and inheritance. We do not introduce new objects with stateful behavior that must be learnt. Everything operates directly on arrays/tensors. This allows users to implement their own more sophisticated objects using these classes as back-end mathematical implementations.

## Installation
Begin by cloning this repo somewhere. To install, go to the clone directory and run

    pip install -e .

## Documentation

Documentation can be found here: https://decargroup.github.io/pylie

## Example

```python
from pylie import SE3 
import numpy as np

# Random pose
T = SE3.random()

# R^n to group directly (using "Capital" notation)
x = np.array([0.1, 0.2, 0.3, 4, 5, 6])
T = SE3.Exp(x)

# Group to R^n directly
x = SE3.Log(T)

# Wedge, vee
Xi = SE3.wedge(x)
x = SE3.vee(Xi)

# Actual exp/log maps 
T = SE3.exp(Xi)
Xi = SE3.log(T)

# Adjoint matrix representation of group element
A = SE3.adjoint(T)

# Inverse of group element
T_inv = SE3.inverse(T)

# Group left/right jacobians, and their inverses
J_L = SE3.left_jacobian(x)
J_R = SE3.right_jacobian(x)
J_L_inv = SE3.left_jacobian_inv(x)
J_R_inv = SE3.right_jacobian_inv(x)

# ... and more.

```
### Using Numpy/C++/Jax
To explicitly access pure numpy implementations use 

```python 
from pylie.numpy import SO2, SO3, SE2, SE3, SE23
```

To explicitly access classes which internally use C++ use 

```python 
from pylie.cpp import SO3, SE3, SE23
```

To explicitly access Jax implementations use

```python 
from pylie.jax import SE2
```

Currently, only `SO3`, `SE3`, and `SE23` are implemented in C++, with the functions accepting and returning numpy arrays. They are also the default internal implementations when simply using `from pylie import SO3, SE3, SE23`. For the JAX implementation, the return types will be `jax.numpy` arrays. All operations in the jax implementation can be JIT-compiled. 


__For all implementations (jax, numpy, C++), the user API is exactly the same! This means that by changing the import statement the example still works.__


**Note:** functions which output "vectors", such as `SE2.Log(T)` all return a 2D numpy array with dimensions `(n, 1)`.


## Running Tests
If you use VS Code, you should be able to enable the VS Code testing feature using pytest. Otherwise, you can run tests from the command line when inside this folder using

    pytest tests

## Credit to UTIAS's STARS group
Some specific implementations came from the [UTIAS STARS Lie group package.](https://github.com/utiasSTARS/liegroups). We wanted a different API and variable ordering, which led to us making our own package. Eventually, this repo evolved to contain more groups, as well as Jax and C++ implementations.
