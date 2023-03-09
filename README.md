# Pylie - with both JAX and Numpy implementations
An instantiation-free python package for common matrix Lie group operations implemented as __pure static classes__. Using pure static classes keeps the usage extremely simple while still allowing for abstraction and inheritance. We do not introduce new objects with stateful behavior that must be learnt. Everything operates directly on arrays/tensors. This allows users to implement their own more sophisticated objects using these classes as back-end mathematical implementations.

## Installation
Begin by cloning this repo somewhere. To install, go to the clone directory and run

    pip install -e .

## Documentation

Documentation can be found here: https://decargroup.github.io/pylie

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
### Using JAX 
Literally copy the above example, but replace the first line with 

```python 
from pylie.jax import SE2
```

Now all return types will be `jax.numpy` arrays. All operations in the jax implementation can be JIT-compiled. 


**Note:** functions which output "vectors", such as `SE2.Log(T)` all return a 2D numpy array with dimensions `(n, 1)`.


## Running Tests
If you use VS Code, you should be able to enable the VS Code testing feature using pytest. Otherwise, you can run tests from the command line when inside this folder using

    pytest tests

## Credit to UTIAS's STARS group
Some specific implementations came from the [UTIAS STARS Lie group package.](https://github.com/utiasSTARS/liegroups).