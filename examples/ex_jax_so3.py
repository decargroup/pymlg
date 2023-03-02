from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")
from pylie.jax import SO3
from pylie import SO3 as SO3_np
import numpy as np
import timeit

x = np.array([0.1, 0.2, 0.3])
# C = SO3.Exp(x) # run it once, it will compile here.

print("Time for 10000 calls of SO3.Exp(x) with jax:")
jax_time = (timeit.timeit(lambda: SO3.Exp(x), number=10000))
print(jax_time)
print("Time for 10000 calls of SO3.Exp(x) without jax:")
np_time = (timeit.timeit(lambda: SO3_np.Exp(x), number=10000))
print(np_time)
print(f"Speedup: {np_time/jax_time:.2f}x")

# Wedge, vee
Xi = SO3.wedge(x)
x = SO3.vee(Xi)

# Actual exp/log maps 
C = SO3.exp(Xi)
Xi = SO3.log(C)

# Adjoint representation of group element
A = SO3.adjoint(C)

# Group left/right jacobians
J_L = SO3.left_jacobian(x)
J_R = SO3.right_jacobian(x)
J_L_inv = SO3.left_jacobian_inv(x)
J_R_inv = SO3.right_jacobian_inv(x)


print(C)