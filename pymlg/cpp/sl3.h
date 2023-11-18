#ifndef _SL3_H_
#define _SL3_H_

#include <Eigen/Dense>
#include <cmath>
#include "base.h"

class SL3 : public MatrixLieGroup<3, 8> {
 public:
  static Element random() { return Exp(Vector::Random()); }

  /**
   * @brief Exponential map: converts a rotation vector to an element of SO(3)
   *
   * @param x rotation vector
   * @return Eigen::Matrix3d
   */
  static Eigen::Matrix3d Exp(const Vector& x) {
    const Eigen::Matrix3d Xi = wedge(x);
    return Xi.exp();
  };

  /**
   * @brief Logarithm map: converts an element of SO(3) to a rotation vector
   *
   * @param x Element of SL3 as 3x3 eigen matrix
   * @return Vector
   */
  static Vector Log(const Element& x) {
    Vector xi;
    xi = SL3::vee(x.log());
    return xi;
  };

  static Element wedge(const Vector& x) {
    Element X;
    // clang-format off
    X <<     x(3)+x(4), -x(2)+x(5),  x(0),
          x(2)+x(5),     x(3)-x(4), x(1), 
         x(6),  x(7),     -2.0*x(3);
    // clang-format on
    return X;
  };

  static Vector vee(const Element& x) {
    Vector xi;
    double half = 1.0/2.0;
    xi << x(0, 2), x(1, 2), half*(x(1, 0)-x(0,1)), -half*x(2,2), half*(x(0, 0)-x(1,1)), half*(x(1, 0)+x(0,1)), x(2,0), x(2,1);
    return xi;
  };

  static Eigen::Matrix<double, 8, 8> leftJacobian(const Vector& xi) {
    Eigen::Matrix<double, 8, 8> J = Eigen::Matrix<double, 8, 8>::Zero(SL3::dof, SL3::dof);

    Element X = SL3::Exp(xi);
    Element exp_inv = SL3::inverse(X);
    //Eigen::Matrix3d J_fd = np.zeros((SL3.dof, SL3.dof))
    double h = 1e-8;
    Vector dx;
    for (int i =0;i<SL3::dof; i++ ){
        dx = Vector::Zero(SL3::dof);
        dx(i) = h;
        J.col(i) = SL3::Log(SL3::Exp(xi + dx)* exp_inv) / h;
    }
    return J;
  };

  static Eigen::Matrix<double,8,8> leftJacobianInverse(const Vector& x) {
    return SL3::leftJacobian(x).inverse();
  };

  static Element inverse(const Element& x) { return x.inverse(); };

  static Eigen::Matrix<double,8,8> adjoint_algebra(const Element& Xi) {

    Vector xi = SL3::vee(Xi);
    Eigen::Matrix<double,8,8> adj;
    adj <<      3.0 * xi(3) + xi(4), -(xi(2) - xi(5)),xi(1),-3.0 * xi(0),-xi(0),-xi(1),0,0,
                xi(2) + xi(5),3.0 * xi(3) - xi(4),-xi(0),-3.0 * xi(1),xi(1),-xi(0),0,0,
                xi(7) / 2.0,-xi(6) / 2.0,0,0,2 * xi(5),-2 * xi(4),xi(1) / 2.0,-xi(0) / 2.0,
                -xi(6) / 2.0, -xi(7) / 2.0, 0, 0, 0, 0, xi(0) / 2.0, xi(1) / 2.0,
                -xi(6) / 2.0,xi(7) / 2.0, 2 * xi(5),0,0,-2 * xi(2),xi(0) / 2.0,-xi(1) / 2.0,
                -xi(7) / 2.0,-xi(6) / 2.0,-2 * xi(4),0,2 * xi(2),0, xi(1) / 2.0,xi(0) / 2.0,
                0,0,xi(7),3.0 * xi(6),xi(6),xi(7),-3.0 * xi(3) - xi(4),-(xi(2) + xi(5)),
                0, 0,-xi(6),3.0 * xi(7),-xi(7),xi(6),xi(2) - xi(5),-(3.0 * xi(3) - xi(4));

    return adj;
   };

  static Eigen::Matrix<double,8,8> adjoint(const Element& x) { 
    Eigen::Matrix<double,9,8> alg;
    Eigen::Matrix<double,8,9> alg_inv;
    Element H_inv_T;
    Eigen::Matrix<double,9,9>C_H;
    Eigen::Matrix<double,8,8> Adj;
    alg <<      0, 0, 0, 1, 1, 0, 0, 0,
                0, 0, -1, 0, 0, 1, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 1, 0, 0,
                0, 0, 0, 1, -1, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, -2, 0, 0, 0, 0;

    alg_inv <<  0, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, -1 / 2.0, 0, 1 / 2.0, 0, 0, 0, 0, 0,
                1 / 2.0, 0, 0, 0, 1 / 2.0, 0, 0, 0, 0,
                1 / 2.0, 0, 0, 0, -1 / 2.0, 0, 0, 0, 0,
                0, 1 / 2.0, 0, 1 / 2.0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0;
    H_inv_T = SL3::inverse(x).transpose();
    C_H  <<     x(0, 0) * H_inv_T, x(0, 1) * H_inv_T, x(0, 2) * H_inv_T,
                x(1, 0) * H_inv_T, x(1, 1) * H_inv_T, x(1, 2) * H_inv_T,
                x(2, 0) * H_inv_T, x(2, 1) * H_inv_T, x(2, 2) * H_inv_T;

    Adj = alg_inv * C_H * alg;


    return Adj;
   };

  static Eigen::Matrix<double,3,8> odot(const Eigen::Vector3d& x) { 
    Eigen::Matrix<double,3,8> p;
    p <<        x(2), 0, -x(1), x(0), x(0), x(1), 0, 0,
                0, x(2), x(0), x(1), -x(1), x(0), 0, 0,
                0, 0, 0, -2 * x(2), 0, 0, x(0), x(1);
            
    return p;

  };

};

#endif