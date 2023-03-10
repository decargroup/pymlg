#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "so3.h"
#include <unsupported/Eigen/MatrixFunctions>


TEST(SO3Test, random_){
  Eigen::Matrix3d m1 = SO3::random();
  EXPECT_TRUE((m1 * m1.transpose()).isApprox(Eigen::Matrix3d::Identity()));
  EXPECT_FLOAT_EQ(m1.determinant(), 1.0);
};


TEST(SO3Test, wedge){

  Eigen::Vector3d v1{1, 2, 3};
  Eigen::Matrix3d m1 = SO3::wedge(v1);
  EXPECT_EQ(m1, -m1.transpose());
};

TEST(SO3Test, properties){
  EXPECT_EQ(SO3::dof, 3);
  EXPECT_EQ(SO3::matrix_size, 3);
};

TEST(SO3Test, vee){
  Eigen::Vector3d v1{1, 2, 3};
  Eigen::Matrix3d m1 = SO3::wedge(v1);
  Eigen::Vector3d v2 = SO3::vee(m1);
  EXPECT_EQ(v1, v2);
};

TEST(SO3Test, Exp){
  Eigen::Vector3d v1{1, 2, 3};
  Eigen::Matrix3d m1 = SO3::Exp(v1);
  EXPECT_TRUE((m1 * m1.transpose()).isApprox(Eigen::Matrix3d::Identity()));
  EXPECT_FLOAT_EQ(m1.determinant(), 1.0);
  Eigen::Matrix3d m2 = SO3::wedge(v1).exp();
  EXPECT_TRUE(m1.isApprox(m2));
};

TEST(SO3Test, Log){
  Eigen::Vector3d v1{0.1, 0.2, 0.3};
  Eigen::Matrix3d m1 = SO3::Exp(v1);
  Eigen::Vector3d v2 = SO3::Log(m1);
  EXPECT_TRUE(v1.isApprox(v2));
};

TEST(SO3Test, leftJacobian){
  Eigen::Vector3d v1{0.1, 0.2, 0.3};
  Eigen::Matrix3d J = SO3::leftJacobian(v1);
  Eigen::Matrix3d Jinv = SO3::leftJacobianInverse(v1);
  EXPECT_TRUE(J.isApprox(Jinv.inverse()));
  
};