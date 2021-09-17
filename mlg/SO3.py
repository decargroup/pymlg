from mlg import MatrixLieGroup
import numpy as np

class SO3(MatrixLieGroup):
    """
    A bare minimum implementation of the SO3 Lie group.
    """
    @staticmethod
    def wedge(x):
        x.reshape(-1,1)
        X = np.array([ 
            [      0, -x[2,0],  x[1,0]],
            [ x[2,0],       0, -x[0,0]],
            [-x[1,0],  x[0,0],       0]
        ])
        return X 
    
    @staticmethod
    def vee(X):
        x = np.array([ 
            [-X[1,2]],
            [ X[0,2]],
            [-X[0,1]]
        ])

if __name__ == "__main__":
    # Quick demo to show how to create an element
    phi = np.array([ 
        [0.1],[0.2],[0.3]
    ])
    
    C = SO3.Exp(phi)

    print(C)