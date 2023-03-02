from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")
from pylie.jax import SE3
from pylie import SE3 as SE3_np
import numpy as np
import timeit

x = np.array([0.1, 0.2, 0.3,4,5,6])
# C = SE3.Exp(x) # run it once, it will compile here.

print("Time for 10000 calls of SE3.Exp(x) with jax:")
jax_time = (timeit.timeit(lambda: SE3.Exp(x), number=10000))
print(jax_time)
print("Time for 10000 calls of SE3.Exp(x) without jax:")
np_time = (timeit.timeit(lambda: SE3_np.Exp(x), number=10000))
print(np_time)
print(f"Speedup: {np_time/jax_time:.2f}x")

# Wedge, vee
Xi = SE3.wedge(x)
x = SE3.vee(Xi)

# Actual exp/log maps 
C = SE3.exp(Xi)
Xi = SE3.log(C)

# Adjoint representation of group element
A = SE3.adjoint(C)

# Group left/right jacobians
J_L = SE3.left_jacobian(x)
J_R = SE3.right_jacobian(x)
J_L_inv = SE3.left_jacobian_inv(x)
J_R_inv = SE3.right_jacobian_inv(x)


print(C)