from .base import MatrixLieGroup, fast_vector_norm
import numpy as np

try:
    # We do not want to make ROS a hard dependency, so we import it only if
    # available.
    from geometry_msgs.msg import Quaternion
except ImportError:
    pass  # ROS is not installed
except:
    raise


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
        xi = np.array(xi).ravel()
        X = np.array(
            [
                [0, -xi[2], xi[1]],
                [xi[2], 0, -xi[0]],
                [-xi[1], xi[0], 0],
            ]
        )
        return X

    @staticmethod
    def cross(xi):
        """
        Alternate name for `SO3.wedge`
        """
        return SO3.wedge(xi)

    @staticmethod
    def vee(X):
        xi = np.array([[-X[1, 2]], [X[0, 2]], [-X[0, 1]]])
        return xi

    @staticmethod
    def exp(Xi):
        """
        Maps elements of the matrix Lie algebra so(3) to the group.

        From Section 8.3 of Lie Groups for Computer Vision by Ethan Eade. When
        theta is small, use Taylor series expansion given in Section 11.
        """
        return SO3.Exp(SO3.vee(Xi))
       

    @staticmethod
    def Exp(xi):
        """
        Maps elements of the vector Lie algebra so(3) to the group.
        """
        phi = np.array(xi).ravel()
        angle = fast_vector_norm(phi)

        # Use Taylor series expansion
        if angle < SO3._small_angle_tol:
            t2 = angle**2
            A = 1.0 - t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0))
            B = (
                1.0
                / 2.0
                * (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0)))
            )
        else:
            A = np.sin(angle) / angle
            B = (1.0 - np.cos(angle)) / (angle**2)

        # Rodirgues rotation formula (103)
        Xi = SO3.wedge(phi)
        return np.eye(3) + A * Xi + B * Xi.dot(Xi)

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
        """
        Computes the Left Jacobian of SO(3).
        From Section 9.3 of Lie Groups for Computer Vision by Ethan Eade.  When
        angle is small, use Taylor series expansion given in Section 11.
        """
        xi = np.array(xi).ravel()
        angle = fast_vector_norm(xi)

        if angle < SO3._small_angle_tol:
            t2 = angle**2
            # Taylor series expansion.  See (157), (159).
            A = (1.0 / 2.0) * (
                1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0))
            )
            B = (1.0 / 6.0) * (
                1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0))
            )
        else:
            A = (1 - np.cos(angle)) / (angle**2)
            B = (angle - np.sin(angle)) / (angle**3)

        cross_xi = SO3.cross(xi)

        J_left = np.eye(3) + A * cross_xi + B * cross_xi.dot(cross_xi)
        return J_left

    @staticmethod
    def left_jacobian_inv(xi):
        """
        Computes the inverse of the left Jacobian of SO(3).
        From Section 9.3 of Lie Groups for Computer Vision by Ethan Eade. When
        angle is small, use Taylor series expansion given in Section 11.
        """
        xi = np.array(xi).ravel()
        angle = fast_vector_norm(xi)
        if angle < SO3._small_angle_tol:
            t2 = angle**2

            # Taylor Series expansion
            A = (1.0 / 12.0) * (
                1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0))
            )
        else:
            A = (1.0 / angle**2) * (
                1.0 - (angle * np.sin(angle) / (2.0 * (1.0 - np.cos(angle))))
            )

        cross_xi = SO3.cross(xi)
        J_left_inv = np.eye(3) - 0.5 * cross_xi + A * cross_xi.dot(cross_xi)

        return J_left_inv

    @staticmethod
    def odot(xi):
        return -SO3.wedge(xi)

    @staticmethod
    def adjoint(C):
        return C

    @staticmethod
    def from_euler(angles, order=[3, 2, 1]):
        """
        Creates a DCM from a 3-element vector of euler angles with specified
        order.

        Parameters
        ----------
        angles : list[float] or ndarray of size 3
            euler angle values
        order : list[int], optional
            euler angle sequence. For example, the default order=[3,2,1] rotates
            the third axis, followed by the second, followed by the first.

        Returns
        -------
        ndarray with shape (3,3)
            DCM corresponding to euler angles
        """

        C = np.identity(3)
        angles = np.array(angles).ravel()

        for i in range(3):
            idx = order[i] - 1
            phi = np.zeros(3)
            phi[idx] = angles[idx]
            C = np.dot(SO3.Exp(phi), C)

        return C

    @staticmethod
    def from_quat(q, order="wxyz"):
        """
        Returns the DCM corresponding to the quaternion representation q.

        .. math::
            \mathbf{C} = (1 - 2 \mathbf{\epsilon}^T \mathbf{\epsilon}) \mathbf{1} \\
            + 2 \mathbf{\epsilon \epsilon}^T + 2 \eta \mathbf{\epsilon}^{\\times}

        Note that the final term is positive to abide by robotics convention. 
        This differs from Barfoot (2019).
        
        Parameters
        ----------
        q : list[float] or ndarray of size 4
            quaternion
        order : str, optional
            quaternion element order "xyzw" or "wxyz", by default "wxyz"

        Returns
        -------
        ndarray with shape (3,3)
            DCM corresponding to `q`

        Raises
        ------
        ValueError
            if `q` is not of size 4
        ValueError
            if `order` is not "xyzw" or "wxyz"
        """

        q = np.array(q).ravel()
        q = q / fast_vector_norm(q)

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
            + 2 * eta * SO3.wedge(eps)
        )

    @staticmethod
    def to_quat(C, order="wxyz"):
        """
        Returns the quaternion corresponding to DCM C.

        Parameters
        ----------
        C : ndarray with shape (3,3)
            DCM/rotation matrix to convert.
        order : str, optional
            quaternion element order "xyzw" or "wxyz", by default "wxyz"

        Returns
        -------
        ndarray with shape (4,1)
             quaternion representation of C

        Raises
        ------
        ValueError
            if `C` does not have shape (3,3)
        ValueError
            if `order` is not "xyzw" or "wxyz"
        """

        C = C.reshape((3, 3))
        if C.shape != (3, 3):
            raise ValueError("C must have shape (3,3).")

        eta = 0.5 * (np.trace(C) + 1) ** 0.5
        eps = -np.array(
            [C[1, 2] - C[2, 1], C[2, 0] - C[0, 2], C[0, 1] - C[1, 0]]
        ) / (4 * eta)

        if order == "wxyz":
            q = np.hstack((eta, eps)).reshape((-1, 1))
        elif order == "xyzw":
            q = np.hstack((eps, eta)).reshape((-1, 1))
        else:
            raise ValueError("order must be 'wxyz' or 'xyzw'. ")

        return q

    @staticmethod
    def to_euler(C):
        """
        Convert a rotation matrix to RPY Euler angles
        :math:`(\\alpha, \\beta, \\gamma)` corresponding to a (3,2,1)
        Euler-angle sequence.
        """
        pitch = np.arctan2(-C[2, 0], np.sqrt(C[0, 0] ** 2 + C[1, 0] ** 2))

        if np.isclose(pitch, np.pi / 2.0):
            yaw = 0.0
            roll = np.arctan2(C[0, 1], C[1, 1])
        elif np.isclose(pitch, -np.pi / 2.0):
            yaw = 0.0
            roll = -np.arctan2(C[0, 1], C[1, 1])
        else:
            sec_pitch = 1.0 / np.cos(pitch)
            yaw = np.arctan2(C[1, 0] * sec_pitch, C[0, 0] * sec_pitch)
            roll = np.arctan2(C[2, 1] * sec_pitch, C[2, 2] * sec_pitch)

        return np.array([roll, pitch, yaw]).ravel()

    @staticmethod
    def from_ros(q):
        """
        Converts a ROS quaternion to a DCM.

        Parameters
        ----------
        q : geometry_msgs.msg.Quaternion
            ROS quaternion

        Returns
        -------
        ndarray with shape (3,3)
            DCM corresponding to `q`
        """
        q = np.array([q.x, q.y, q.z, q.w])
        return SO3.from_quat(q, order="xyzw")

    @staticmethod
    def to_ros(C):
        """
        Converts a DCM to a ROS quaternion.

        Parameters
        ----------
        C : ndarray with shape (3,3)
            DCM to convert

        Returns
        -------
        geometry_msgs.msg.Quaternion
            ROS quaternion corresponding to `C`
        """
        q = SO3.to_quat(C, order="wxyz").ravel()
        msg = Quaternion()
        msg.w = q[0]
        msg.x = q[1]
        msg.y = q[2]
        msg.z = q[3]
        return msg

    @staticmethod
    def identity():
        """
        Returns the identity DCM.
        """
        return np.eye(3)
