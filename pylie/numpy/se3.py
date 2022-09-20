from .base import MatrixLieGroup
import numpy as np
from .so3 import SO3

try:
    # We do not want to make ROS a hard dependency, so we import it only if
    # available.
    from geometry_msgs.msg import Pose
except ImportError:
    pass  # ROS is not installed
except:
    raise

class SE3(MatrixLieGroup):
    """
    An instantiation-free implementation of the SE3 matrix Lie group.
    """

    dof = 6

    @staticmethod
    def synthesize(C, r):
        """
        Deprecated. Use `SE3.from_components(C,r)` instead.

        Construct an SE(3) matrix from a rotation matrix and translation vector.
        """
        return SE3.from_components(C, r)

    @staticmethod
    def from_components(C, r):
        """
        Construct an SE(3) matrix from a rotation matrix and translation vector.
        """
        # Check if rotation component is a rotation vector or full DCM
        if C.size == 9:
            C = C
        else:
            C = SO3.Exp(C)

        r = np.array(r).reshape((-1, 1))

        return np.block([[C, r], [np.zeros((1, 3)), 1]])

    @staticmethod
    def to_components(T):
        """
        Decompose an SE(3) matrix into a rotation matrix and translation vector.
        """
        C = T[0:3, 0:3]
        r = T[0:3, 3]
        return C, r

    @staticmethod
    def from_ros(pose_msg):
        """
        Constructs an SE(3) matrix from a ROS Pose message.

        Parameters
        ----------
        pose_msg: geometry_msgs.msg.Pose
            ROS Pose message from geometry_msgs

        Returns
        -------
        np.ndarray with shape (4,4)
            SE(3) pose transformation matrix
        """
        q = pose_msg.orientation
        pos = pose_msg.position
        C = SO3.from_quat([q.w, q.x, q.y, q.z], order="wxyz")
        r = np.array([pos.x, pos.y, pos.z])
        return SE3.from_components(C, r)

    @staticmethod 
    def to_ros(T):
        """
        Constructs a ROS Pose message from an SE(3) matrix.

        Parameters
        ----------
        T : np.ndarray with shape (4,4)
            SE(3) pose transformation matrix

        Returns
        -------
        geometry_msgs.msg.Pose
            ROS Pose message
        """
        C, r = SE3.to_components(T)
        r = r.ravel()
        q = SO3.to_quat(C, order="wxyz").ravel()
        msg = Pose()
        msg.position.x = r[0]
        msg.position.y = r[1]
        msg.position.z = r[2]
        msg.orientation.w = q[0]
        msg.orientation.x = q[1]
        msg.orientation.y = q[2]
        msg.orientation.z = q[3]
        return msg         


    @staticmethod
    def random():
        phi = np.random.uniform(0, 2 * np.pi, (3, 1))
        r = np.random.normal(0, 1, (3, 1))
        C = SO3.Exp(phi)
        T = np.block([[C, r], [np.zeros((1, 3)), 1]])
        return T

    @staticmethod
    def wedge(xi):
        xi = np.array(xi).ravel()
        phi = xi[0:3]
        xi_r = xi[3:]
        Xi_phi = SO3.wedge(phi)
        return np.block([[Xi_phi, xi_r.reshape((-1, 1))], [np.zeros((1, 4))]])

    @staticmethod
    def vee(Xi):
        Xi_phi = Xi[0:3, 0:3]
        xi_r = Xi[0:3, 3]
        phi = SO3.vee(Xi_phi)
        return np.vstack((phi, xi_r.reshape((-1, 1))))

    @staticmethod
    def exp(Xi):
        Xi_phi = Xi[0:3, 0:3]
        phi = SO3.vee(Xi_phi)
        xi_r = Xi[0:3, 3]
        C = SO3.exp(Xi_phi)
        r = np.dot(SO3.left_jacobian(phi), xi_r.reshape((-1, 1)))
        T = np.block([[C, r], [np.zeros((1, 3)), 1]])
        return T

    @staticmethod
    def log(T):
        Xi_phi = SO3.log(T[0:3, 0:3])
        r = T[0:3, 3]
        xi_r = np.dot(SO3.left_jacobian_inv(SO3.vee(Xi_phi)), r.reshape((-1, 1)))
        Xi = np.block([[Xi_phi, xi_r], [np.zeros((1, 4))]])
        return Xi

    @staticmethod
    def odot(b):
        b = np.array(b).ravel()
        return np.block(
            [
                [SO3.odot(b[0:3]), b[3] * np.identity(3)],
                [np.zeros((1, 3)), np.zeros((1, 3))],
            ]
        )

    @staticmethod
    def adjoint(T):
        C = T[0:3, 0:3]
        r = T[0:3, 3]
        return np.block([[C, np.zeros((3, 3))], [np.dot(SO3.wedge(r), C), C]])

    @staticmethod
    def _left_jacobian_Q_matrix(xi):
        xi = np.array(xi).ravel()

        if xi.size != SE3.dof:
            raise ValueError("xi must have length {}".format(SE3.dof))

        phi = xi[0:3]  # rotation part
        rho = xi[3:6]  # translation part

        rx = SO3.wedge(rho)
        px = SO3.wedge(phi)

        ph = np.linalg.norm(phi)
        if ph < SE3._small_angle_tol:
            l = 1
        ph2 = ph * ph
        ph3 = ph2 * ph
        ph4 = ph3 * ph
        ph5 = ph4 * ph

        cph = np.cos(ph)
        sph = np.sin(ph)

        m1 = 0.5
        m2 = (ph - sph) / ph3
        m3 = (0.5 * ph2 + cph - 1.0) / ph4
        m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

        t1 = rx
        t2 = px.dot(rx) + rx.dot(px) + px.dot(rx).dot(px)
        t3 = px.dot(px).dot(rx) + rx.dot(px).dot(px) - 3.0 * px.dot(rx).dot(px)
        t4 = px.dot(rx).dot(px).dot(px) + px.dot(px).dot(rx).dot(px)

        return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

    @staticmethod
    def left_jacobian(xi):

        xi = np.array(xi).ravel()
        if np.linalg.norm(xi) < SE3._small_angle_tol:
            return np.identity(6)

        else:
            Q = SE3._left_jacobian_Q_matrix(xi)

            phi = xi[0:3]  # rotation part

            J = SO3.left_jacobian(phi)
            return np.block([[J, np.zeros((3, 3))], [Q, J]])

    @staticmethod
    def left_jacobian_inv(xi):
        xi = np.array(xi).ravel()

        if np.linalg.norm(xi) < SE3._small_angle_tol:
            return np.identity(6)

        else:
            Q = SE3._left_jacobian_Q_matrix(xi)

            phi = xi[0:3]  # rotation part

            J_inv = SO3.left_jacobian_inv(phi)

            return np.block(
                [[J_inv, np.zeros((3, 3))], [-np.dot(J_inv, np.dot(Q, J_inv)), J_inv]]
            )

    @staticmethod
    def identity():
        return np.identity(4)