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
T = SE3.Exp([0.1, 0.2, 0.3, 4, 5, 6])

# Group to R^n directly
x = SE3.Log(T)

# Wedge, vee
Xi = SE3.wedge(x)
x = SE3.vee(x)

# Actual exp/log maps 
T = SE3.exp(Xi)
Xi = SE3.log(T)

```