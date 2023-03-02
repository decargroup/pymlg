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
