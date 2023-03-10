#ifndef _SE3_H_
#define _SE3_H_

#include <Eigen/Dense>
#include "base.h"
#include "so3.h"

class SE3 : public MatrixLieGroup<4, 6> {
 public:
  static Element random() { return Exp(Vector::Random()); }
  /**
   * @brief Exponential map: converts exponential coordinates to an element of
   *  SE(3)
   *
   * @param x twist
   * @return Eigen::Matrix4d
   */
  static Eigen::Matrix4d Exp(const Vector& x) {
    Eigen::Matrix4d X = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d R = SO3::Exp(x.block<3, 1>(0, 0));
    Eigen::Vector3d xi_r = x.block<3, 1>(3, 0);
    X.block<3, 3>(0, 0) = R;
    X.block<3, 1>(0, 3) = SO3::leftJacobian(x.block<3, 1>(0, 0)) * xi_r;
    return X;
  };

  /**
   * @brief Logarithm map: converts an element of SE(3) to  exponential
   * coordinates
   *
   * @param x Element of SE3 as 4x4 eigen matrix
   * @return Vector
   */
  static Vector Log(const Eigen::Matrix4d& x) {
    Vector xi;
    Eigen::Matrix3d R = x.block<3, 3>(0, 0);
    Eigen::Vector3d p = x.block<3, 1>(0, 3);
    xi.block<3, 1>(0, 0) = SO3::Log(R);
    xi.block<3, 1>(3, 0) = SO3::leftJacobianInverse(xi.block<3, 1>(0, 0)) * p;
    return xi;
  };

  /**
   * @brief Converts exponential coordinates to 4x4 lie algebra matrix
   *
   * @param x 6x1 vector of exponential coordinates
   * @return Eigen::Matrix4d  4x4 Lie algebra matrix
   */
  static Eigen::Matrix4d wedge(const Vector& x) {
    Eigen::Matrix4d X;
    // clang-format off
    X << SO3::wedge(x.block<3, 1>(0, 0)), x.block<3, 1>(3, 0),
         0, 0, 0, 0;
    // clang-format on
    return X;
  };

  /**
   * @brief Converts 4x4 lie algebra matrix to exponential coordinates
   *
   * @param x 4x4 Lie algebra matrix
   * @return Vector exponential coordinates
   */
  static Vector vee(const Eigen::Matrix4d& x) {
    Vector xi;
    xi.block<3, 1>(0, 0) = SO3::vee(x.block<3, 3>(0, 0));
    xi.block<3, 1>(3, 0) = x.block<3, 1>(0, 3);
    return xi;
  };

  static Eigen::Matrix3d leftJacobianQMatrix(const Eigen::Vector3d& phi,
                                                         const Eigen::Vector3d& xi_r) {
    Eigen::Matrix3d rx{SO3::wedge(xi_r)};
    Eigen::Matrix3d px{SO3::wedge(phi)};

    double ph{phi.norm()};

    double ph2{ph * ph};
    double ph3{ph2 * ph};
    double ph4{ph3 * ph};
    double ph5{ph4 * ph};

    double cph{cos(ph)};
    double sph{sin(ph)};

    double m1{0.5};
    double m2{(ph - sph) / ph3};
    double m3{(0.5 * ph2 + cph - 1.0) / ph4};
    double m4{(ph - 1.5 * sph + 0.5 * ph * cph) / ph5};

    Eigen::Matrix3d pxrx{px * rx};
    Eigen::Matrix3d rxpx{rx * px};
    Eigen::Matrix3d pxrxpx{pxrx * px};

    Eigen::Matrix3d t1{rx};
    Eigen::Matrix3d t2{pxrx + rxpx + pxrxpx};
    Eigen::Matrix3d t3{px * pxrx + rxpx * px - 3.0 * pxrxpx};
    Eigen::Matrix3d t4{pxrxpx * px + px * pxrxpx};

    return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4;
  };

  static Eigen::Matrix<double, 6, 6> leftJacobian(const Vector& x) {
    Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix3d Jso3{SO3::leftJacobian(x.block<3, 1>(0, 0))};
    J.block<3, 3>(0, 0) = Jso3;
    J.block<3, 3>(3, 0) = SE3::leftJacobianQMatrix(x.block<3, 1>(0, 0),
                                                   x.block<3, 1>(3, 0));
    J.block<3, 3>(3, 3) = Jso3;
    return J;
  };

  static Eigen::Matrix<double, 6, 6> leftJacobianInverse(const Vector& x){
    Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix3d Jinv{SO3::leftJacobianInverse(x.block<3, 1>(0, 0))};
    Eigen::Matrix3d Q{SE3::leftJacobianQMatrix(x.block<3, 1>(0, 0),
                                                   x.block<3, 1>(3, 0))};
    J.block<3, 3>(0, 0) = Jinv;
    J.block<3, 3>(3, 0) = -Jinv * Q * Jinv;
    J.block<3, 3>(3, 3) = Jinv;
    return J;
  };

  static Eigen::Matrix4d inverse(const Eigen::Matrix4d& x) {
    Eigen::Matrix4d Xinv = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d R = x.block<3, 3>(0, 0);
    Eigen::Vector3d p = x.block<3, 1>(0, 3);
    Xinv.block<3, 3>(0, 0) = R.transpose();
    Xinv.block<3, 1>(0, 3) = -R.transpose() * p;
    return Xinv;
  };

  static Eigen::Matrix<double, 6, 6> adjoint(const Eigen::Matrix4d& T) {
    Eigen::Matrix<double, 6, 6>  Xadj;
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d p = T.block<3, 1>(0, 3);
    Xadj.block<3, 3>(0, 0) = R;
    Xadj.block<3, 3>(0, 3) = SO3::wedge(p) * R;
    Xadj.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
    Xadj.block<3, 3>(3, 3) = R;

  };
};
#endif