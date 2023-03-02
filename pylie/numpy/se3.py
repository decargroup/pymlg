from .base import MatrixLieGroup, fast_vector_norm
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
    matrix_size = 4

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
        T = np.zeros((4, 4))
        T[0:3, 0:3] = C
        T[0:3, 3] = r.ravel()
        T[3, 3] = 1
        return T

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
        phi = np.random.uniform(-np.pi, np.pi, (3,))
        r = np.random.normal(0, 1, (3, 1))
        C = SO3.Exp(phi)
        return SE3.from_components(C, r)

    @staticmethod
    def wedge(xi):
        xi = np.array(xi).ravel()
        phi = xi[0:3]
        xi_r = xi[3:]
        Xi_phi = SO3.wedge(phi)
        Xi = np.zeros((4, 4))
        Xi[0:3, 0:3] = Xi_phi
        Xi[0:3, 3] = xi_r
        return Xi

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
        return SE3.from_components(C, r)

    @staticmethod
    def log(T):
        Xi_phi = SO3.log(T[0:3, 0:3])
        r = T[0:3, 3]
        xi_r = np.dot(
            SO3.left_jacobian_inv(SO3.vee(Xi_phi)), r.reshape((-1, 1))
        )
        Xi = np.zeros((4, 4))
        Xi[0:3, 0:3] = Xi_phi
        Xi[0:3, 3] = xi_r.ravel()
        return Xi

    @staticmethod
    def odot(b):
        b = np.array(b).ravel()
        X = np.zeros((4, 6))
        X[0:3, 0:3] = SO3.odot(b[0:3])
        X[0:3, 3:6] = b[3] * np.identity(3)
        return X

    @staticmethod
    def adjoint(T):
        C = T[0:3, 0:3]
        r = T[0:3, 3]
        Ad = np.zeros((6, 6))
        Ad[0:3, 0:3] = C
        Ad[3:6, 3:6] = C
        Ad[3:6, 0:3] = np.dot(SO3.wedge(r), C)
        return Ad

    @staticmethod
    def _left_jacobian_Q_matrix(phi, rho):
        phi = np.array(phi).ravel()
        rho = np.array(rho).ravel()

        rx = SO3.wedge(rho)
        px = SO3.wedge(phi)

        ph = fast_vector_norm(phi)

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

        pxrx = px.dot(rx)
        rxpx = rx.dot(px)
        pxrxpx = pxrx.dot(px)

        t1 = rx
        t2 = pxrx + rxpx + pxrxpx
        t3 = px.dot(pxrx) + rxpx.dot(px) - 3.0 * pxrxpx
        t4 = pxrxpx.dot(px) + px.dot(pxrxpx)

        return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

    @staticmethod
    def left_jacobian(xi):

        xi = np.array(xi).ravel()

        phi = xi[0:3]  # rotation part
        rho = xi[3:6]  # translation part

        if fast_vector_norm(phi) < SE3._small_angle_tol:
            return np.identity(6)

        else:
            Q = SE3._left_jacobian_Q_matrix(phi, rho)

            phi = xi[0:3]  # rotation part

            J = SO3.left_jacobian(phi)
            out = np.zeros((6, 6))
            out[0:3, 0:3] = J
            out[3:6, 3:6] = J
            out[3:6, 0:3] = Q
            return out

    @staticmethod
    def left_jacobian_inv(xi):
        xi = np.array(xi).ravel()

        if fast_vector_norm(xi) < SE3._small_angle_tol:
            return np.identity(6)

        else:

            phi = xi[0:3]  # rotation part
            rho = xi[3:6]  # translation part
            Q = SE3._left_jacobian_Q_matrix(phi, rho)

            J_inv = SO3.left_jacobian_inv(phi)

            out = np.zeros((6, 6))
            out[0:3, 0:3] = J_inv
            out[3:6, 3:6] = J_inv
            out[3:6, 0:3] = -np.dot(J_inv, np.dot(Q, J_inv))
            return out
