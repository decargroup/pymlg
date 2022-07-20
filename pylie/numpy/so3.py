from .base import MatrixLieGroup
import numpy as np


class SO3(MatrixLieGroup):
    """
    An instantiation-free implementation of the SO3 matrix Lie group.
    """

    dof = 3

    @staticmethod
    def random():
        v = np.random.uniform(0, 2 * np.pi, (3, 1))
        return SO3.Exp(v)

    @staticmethod
    def wedge(xi):
        xi = np.array(xi).reshape((-1, 1))
        X = np.array(
            [
                [0, -xi[2, 0], xi[1, 0]],
                [xi[2, 0], 0, -xi[0, 0]],
                [-xi[1, 0], xi[0, 0], 0],
            ]
        )
        return X

    @staticmethod
    def cross(xi):
        return SO3.wedge(xi)

    @staticmethod
    def vee(X):
        xi = np.array([[-X[1, 2]], [X[0, 2]], [-X[0, 1]]])
        return xi

    @staticmethod
    def exp(element_so3):
        """Maps elements of the matrix Lie algebra so(3) to the group.

        From Section 8.3 of Lie Groups for Computer Vision by Ethan Eade. When
        theta is small, use Taylor series expansion given in Section 11.
        """
        phi = SO3.vee(element_so3)
        angle = np.linalg.norm(phi)

        # Use Taylor series expansion
        if angle < SO3._small_angle_tol:
            t2 = angle**2
            A = 1.0 - t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0))
            B = 1.0 / 2.0 * (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0)))
        else:
            A = np.sin(angle) / angle
            B = (1.0 - np.cos(angle)) / (angle**2)

        # Rodirgues rotation formula (103)
        return np.eye(3) + A * element_so3 + B * np.dot(element_so3, element_so3)

    @staticmethod
    def log(X):
        # The cosine of the rotation angle is related to the trace of C
        cos_angle = 0.5 * np.trace(X) - 0.5
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.isclose(angle, 0.0):
            return X - np.identity(3)

        # Otherwise take the matrix logarithm and return the rotation vector
        return (0.5 * angle / np.sin(angle)) * (X - np.transpose(X))

    @staticmethod
    def left_jacobian(xi):
        """Computes the Left Jacobian of SO(3).
        From Section 9.3 of Lie Groups for Computer Vision by Ethan Eade.  When
        angle is small, use Taylor series expansion given in Section 11.
        """
        angle = np.linalg.norm(xi)

        if angle < SO3._small_angle_tol:
            t2 = angle**2
            # Taylor series expansion.  See (157), (159).
            A = (1.0 / 2.0) * (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0)))
            B = (1.0 / 6.0) * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0)))
        else:
            A = (1 - np.cos(angle)) / (angle**2)
            B = (angle - np.sin(angle)) / (angle**3)

        cross_xi = SO3.cross(xi)

        J_left = np.eye(3) + A * cross_xi + B * np.dot(cross_xi, cross_xi)
        return J_left

    @staticmethod
    def left_jacobian_inv(xi):
        """Computes the inverse of the left Jacobian of SO(3).
        From Section 9.3 of Lie Groups for Computer Vision by Ethan Eade. When
        angle is small, use Taylor series expansion given in Section 11.
        """
        angle = np.linalg.norm(xi)
        if angle < SO3._small_angle_tol:
            t2 = angle**2

            # Taylor Series expansion
            A = (1.0 / 12.0) * (1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0)))
        else:
            A = (1.0 / angle**2) * (
                1.0 - (angle * np.sin(angle) / (2.0 * (1.0 - np.cos(angle))))
            )

        cross_xi = SO3.cross(xi)
        J_left_inv = np.eye(3) - 0.5 * cross_xi + A * np.dot(cross_xi, cross_xi)

        return J_left_inv

    @staticmethod
    def odot(xi):
        return -SO3.wedge(xi)

    @staticmethod
    def adjoint(C):
        return C

    @staticmethod 
    def from_euler(theta, order=[3,2,1]):
        """
        Creates a DCM from a 3-element vector of euler angles with specified 
        order.

        PARAMETERS
        ----------
        theta: ndarray of euler angles with shape (3,) or (3,1)
        order: list of integers specifying the order sequence. for example,
            the default order=[3,2,1] rotates the third axis, followed by 
            the second, followed by the first.
        """

        C = np.identity(3)
        theta = theta.ravel()

        for i in range(3):
            idx = order[i] -1
            phi = np.zeros(3)
            phi[idx] = theta[i]
            C = np.dot(SO3.Exp(phi), C) 

        return C

    @staticmethod 
    def from_quat(q, order="wxyz"):
        """
        Returns the DCM corresponding to the quaternion representation q. 

        PARAMETERS
        ----------
        q: list or ndarray of size 4, of unit length
        order: "wxyz" or "xyzw". specifies what each component in q means.
        """
        q = np.array(q).ravel()
        q = q/np.linalg.norm(q)

        if q.size != 4:
            raise ValueError("q must have size 4.")
        if order == "wxyz":
            eta = q[0]
            eps = q[1:]
        elif order == "xyzw":
            eta = q[3]
            eps = q[0:3]
        else:
            raise ValueError("order must be 'wxyz' or 'xyzw'. ")
            

        eps = eps.reshape((-1, 1))

        return (
            (1 - 2 * np.matmul(np.transpose(eps), eps)) * np.eye(3)
            + 2 * np.matmul(eps, np.transpose(eps))
            - 2 * eta * SO3.wedge(eps)
        )

    @staticmethod
    def to_quat(C, order="wxyz"):
        """
        Returns the quaternion corresponding to DCM C. 

        PARAMETERS
        ----------
        C: numpy ndarray with shape (3,3)
        order: "wxyz" or "xyzw". specifies what each component in q means.
        """
        C = C.reshape((3,3))
        if C.shape != (3,3):
            raise ValueError("C must have shape (3,3).")


        eta = 0.5*(np.trace(C) + 1) ** 0.5
        eps = np.array([
            C[1,2] - C[2,1],
            C[2,0] - C[0,2],
            C[0,1] - C[1,0]
        ])/(4*eta)

        if order == "wxyz":
            q = np.hstack((eta, eps)).reshape((-1,1))
        elif order == "xyzw":
            q = np.hstack((eps, eta)).reshape((-1,1))
        else:
            raise ValueError("order must be 'wxyz' or 'xyzw'. ")

        return q