#ifndef _LIE_BASE_H
#define _LIE_BASE_H
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

template <int n_, int dof_>
class MatrixLieGroup {
 public:
  inline static const int dof = dof_;
  inline static const int matrix_size = n_;
  inline static constexpr float small_angle_tol = 1e-7;
  using Element = Eigen::Matrix<double, n_, n_>;
  using Vector = Eigen::Matrix<double, dof_, 1>;

  static Element Exp(const Vector& x);
  static Vector Log(const Element& x);
  static Element exp(const Element& x){ return x.exp(); };
  static Element log(const Element& x){ return x.log(); };
  static Element wedge(const Vector& x);
  static Vector vee(const Element& x);
  static Eigen::Matrix<double, dof_, dof_> leftJacobian(const Vector& x);
  static Eigen::Matrix<double, dof_, dof_> leftJacobianInverse(const Vector& x);

  static Eigen::Matrix<double, dof_, dof_> rightJacobian(const Vector& x) {
    return leftJacobian(-x);
  };
  static Eigen::Matrix<double, dof_, dof_> rightJacobianInverse(
      const Vector& x) {
    return leftJacobianInverse(-x);
  };

  static Element identity() { return Element::Identity(); };

  static Element inverse(const Element& x) { return x.inverse(); };
};

#endif