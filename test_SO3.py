from pylie import SO3
import numpy as np

C = SO3.Exp(np.array([[0],[1],[0]]))

print(C)