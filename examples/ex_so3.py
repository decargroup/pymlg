from pylie.cpp import SO3
from pylie import SO3 as SO3np 
import timeit 
import numpy as np 

x = SO3np.random()

def test_wrapper(x):
    # x = np.array(x, copy=False).ravel()
    return SO3.Log(x).reshape((-1,1))


time_cpp = timeit.timeit(lambda:test_wrapper(x), number=100000)
time_np = timeit.timeit(lambda: SO3np.Log(x), number=100000)

print(f"Time for cpp: {time_cpp}")
print(f"Time for np: {time_np}")
print(f"Ratio: {time_np/time_cpp}")
