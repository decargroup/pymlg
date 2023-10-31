#ifndef _SE23_H_
#define _SE23_H_

#include <Eigen/Dense>

#include "base.h"
#include "se3.h"
#include "so3.h"

class SE23 : public MatrixLieGroup<5, 9> {
 public:
  static Element random() { return Exp(Vector::Random()); }
  /**
   * @brief Exponential map: converts exponential coordinates to an element of
   *  SE2(3)
   *
   * @param x twist
   * @return Element
   */
  static Element Exp(const Vector& x) {
    Element X = Element::Identity();
    Eigen::Matrix3d R{SO3::Exp(x.block<3, 1>(0, 0))};
    Eigen::Vector3d xi_v{x.block<3, 1>(3, 0)};
    Eigen::Vector3d xi_r{x.block<3, 1>(6, 0)};
    Eigen::Matrix3d J{SO3::leftJacobian(x.block<3, 1>(0, 0))};
    X.block<3, 3>(0, 0) = R;
    X.block<3, 1>(0, 3) = J * xi_v;
    X.block<3, 1>(0, 4) = J * xi_r;
    return X;
  };

  /**
   * @brief Logarithm map: converts an element of SE(3) to  exponential
   * coordinates
   *
   * @param x Element of SE3 as 5x5 eigen matrix
   * @return Vector
   */
  static Vector Log(const Element& x) {
    Vector xi;
    Eigen::Matrix3d R{x.block<3, 3>(0, 0)};
    Eigen::Vector3d v{x.block<3, 1>(0, 3)};
    Eigen::Vector3d r{x.block<3, 1>(0, 4)};
    Eigen::Vector3d phi{SO3::Log(R)};
    Eigen::Matrix3d J_inv{SO3::leftJacobianInverse(phi)};
    xi.block<3, 1>(0, 0) = phi;
    xi.block<3, 1>(3, 0) = J_inv * v;
    xi.block<3, 1>(6, 0) = J_inv * r;
    return xi;
  };

  /**
   * @brief Converts exponential coordinates to 5x5 lie algebra matrix
   *
   * @param x 6x1 vector of exponential coordinates
   * @return Element  5x5 Lie algebra matrix
   */
  static Element wedge(const Vector& x) {
    Element X{Element::Zero()};
    X.block<3, 3>(0, 0) = SO3::wedge(x.block<3, 1>(0, 0));
    X.block<3, 1>(0, 3) = x.block<3, 1>(3, 0);
    X.block<3, 1>(0, 4) = x.block<3, 1>(6, 0);
    return X;
  };

  /**
   * @brief Converts 5x5 lie algebra matrix to exponential coordinates
   *
   * @param x 5x5 Lie algebra matrix
   * @return Vector exponential coordinates
   */
  static Vector vee(const Element& x) {
    Vector xi;
    xi.block<3, 1>(0, 0) = SO3::vee(x.block<3, 3>(0, 0));
    xi.block<3, 1>(3, 0) = x.block<3, 1>(0, 3);
    xi.block<3, 1>(6, 0) = x.block<3, 1>(0, 4);
    return xi;
  };

  static Eigen::Matrix<double, 9, 9> leftJacobian(const Vector& x) {
    // Check if rotation component is small
    Eigen::Vector3d phi{x.block<3, 1>(0, 0)};
    if (phi.norm() < SE23::small_angle_tol) {
      return Eigen::Matrix<double, 9, 9>::Identity();
    } else {
      Eigen::Vector3d xi_v{x.block<3, 1>(3, 0)};
      Eigen::Vector3d xi_r{x.block<3, 1>(6, 0)};
      Eigen::Matrix3d Jso3{SO3::leftJacobian(phi)};
      Eigen::Matrix3d Q_v{SE3::leftJacobianQMatrix(phi, xi_v)};
      Eigen::Matrix3d Q_r{SE3::leftJacobianQMatrix(phi, xi_r)};

      Eigen::Matrix<double, 9, 9> J{Eigen::Matrix<double, 9, 9>::Zero()};
      J.block<3, 3>(0, 0) = Jso3;
      J.block<3, 3>(3, 3) = Jso3;
      J.block<3, 3>(6, 6) = Jso3;
      J.block<3, 3>(3, 0) = Q_v;
      J.block<3, 3>(6, 0) = Q_r;
      return J;
    }
  };

  static Eigen::Matrix<double, 9, 9> leftJacobianInverse(const Vector& x) {
    // Check if rotation component is small
    Eigen::Vector3d phi{x.block<3, 1>(0, 0)};
    if (x.block<3, 1>(0, 0).norm() < SE3::small_angle_tol) {
      return Eigen::Matrix<double, 9, 9>::Identity();
    } else {
      Eigen::Vector3d xi_v{x.block<3, 1>(3, 0)};
      Eigen::Vector3d xi_r{x.block<3, 1>(6, 0)};
      Eigen::Matrix3d Jinv{SO3::leftJacobianInverse(x.block<3, 1>(0, 0))};
      Eigen::Matrix3d Q_v{SE3::leftJacobianQMatrix(phi, xi_v)};
      Eigen::Matrix3d Q_r{SE3::leftJacobianQMatrix(phi, xi_r)};

      Eigen::Matrix<double, 9, 9> J = Eigen::Matrix<double, 9, 9>::Zero();
      J.block<3, 3>(0, 0) = Jinv;
      J.block<3, 3>(3, 3) = Jinv;
      J.block<3, 3>(6, 6) = Jinv;
      J.block<3, 3>(3, 0) = -Jinv * Q_v * Jinv;
      J.block<3, 3>(6, 0) = -Jinv * Q_r * Jinv;
      return J;
    }
  };

  static Element inverse(const Element& x) {
    Element Xinv = Element::Identity();
    Eigen::Matrix3d R = x.block<3, 3>(0, 0);
    Eigen::Vector3d v = x.block<3, 1>(0, 3);
    Eigen::Vector3d r = x.block<3, 1>(0, 4);
    Eigen::Matrix3d Rinv = R.transpose();
    Xinv.block<3, 3>(0, 0) = Rinv;
    Xinv.block<3, 1>(0, 3) = -Rinv * v;
    Xinv.block<3, 1>(0, 4) = -Rinv * r;
    return Xinv;
  };

  static Eigen::Matrix<double, 9, 9> adjoint(const Element& T) {
    Eigen::Matrix<double, 9, 9> Xadj{Eigen::Matrix<double, 9, 9>::Zero()};
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d v = T.block<3, 1>(0, 3);
    Eigen::Vector3d r = T.block<3, 1>(0, 4);
    Xadj.block<3, 3>(0, 0) = R;
    Xadj.block<3, 3>(3, 3) = R;
    Xadj.block<3, 3>(6, 6) = R;
    Xadj.block<3, 3>(3, 0) = SO3::wedge(v) * R;
    Xadj.block<3, 3>(6, 0) = SO3::wedge(r) * R;
    return Xadj;
  };

  static Eigen::Matrix<double, 9, 9> adjoint_algebra(const Element& T) {
    Eigen::Matrix<double, 9, 9> adj{Eigen::Matrix<double, 9, 9>::Zero()};
    Eigen::Matrix3d xi_phi_cross = T.block<3, 3>(0, 0);
    Eigen::Vector3d xi_v = T.block<3, 1>(0, 3);
    Eigen::Vector3d xi_r = T.block<3, 1>(0, 4);
    adj.block<3, 3>(0, 0) = xi_phi_cross;
    adj.block<3, 3>(3, 3) = xi_phi_cross;
    adj.block<3, 3>(6, 6) = xi_phi_cross;
    adj.block<3, 3>(3, 0) = SO3::wedge(xi_v);
    adj.block<3, 3>(6, 0) = SO3::wedge(xi_r);
    return adj;
  }

  static Eigen::Matrix<double, 5, 9> odot(const Eigen::Matrix<double, 5, 1>& b) {
    Eigen::Matrix<double, 5, 9> B{ Eigen::Matrix<double, 5, 9>::Zero() };
    B.block<4, 6>(0, 0) = SE3::odot(b.block<4, 1>(0, 0));
    B.block<3, 3>(0, 6) = b(4) * Eigen::Matrix3d::Identity();
    return B;
  };
};
#endif