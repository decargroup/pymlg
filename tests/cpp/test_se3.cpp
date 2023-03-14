
#include<gtest/gtest.h>
#include<Eigen/Dense>
#include <iostream>
#include "se3.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <chrono>

TEST(SE3Test, wedge){
  Eigen::Matrix<double, 6, 1> v1;
  v1 << 1, 2, 3, 4, 5, 6;
  Eigen::Matrix4d m1 = SE3::wedge(v1);
  Eigen::Matrix3d m2 = m1.block(0,0,3,3);
  EXPECT_EQ(m2, -m2.transpose());
};

TEST(SE3Test, vee){
  Eigen::Matrix<double, 6, 1> v1;
  v1 << 1, 2, 3, 4, 5, 6;
  Eigen::Matrix4d m1 = SE3::wedge(v1);
  Eigen::Matrix<double, 6, 1> v2 = SE3::vee(m1);
  EXPECT_EQ(v1, v2);
};

TEST(SE3Test, Exp){
  Eigen::Matrix<double, 6, 1> v1;
  v1 << 0.0,0.0,0.2, 0.4, 0.5, 0.6;
  Eigen::Matrix4d m1 = SE3::Exp(v1);
  Eigen::Matrix4d m3 = (SE3::wedge(v1)).exp();
  Eigen::Matrix3d R = m1.block(0,0,3,3);

  EXPECT_TRUE((R * R.transpose()).isApprox(Eigen::Matrix3d::Identity()));
  EXPECT_FLOAT_EQ(R.determinant(), 1.0);

  EXPECT_TRUE(m1.isApprox(m3));

};

TEST(SE3Test, Log){
  Eigen::Matrix<double, 6, 1> v1;
  v1 << 0.0,0.0,0.2, 0.4, 0.5, 0.6;
  Eigen::Matrix4d m1 = SE3::Exp(v1);
  Eigen::Matrix<double, 6, 1> v2 = SE3::Log(m1);
  EXPECT_TRUE(v1.isApprox(v2));
};

TEST(SE3Test, leftJacobian){
  Eigen::Matrix<double, 6, 1> v1;
  v1 << 0.0,0.0,0.2, 0.4, 0.5, 0.6;
  Eigen::Matrix<double, 6, 6> J = SE3::leftJacobian(v1);
  Eigen::Matrix<double, 6, 6> Jinv = SE3::leftJacobianInverse(v1);
  EXPECT_TRUE(J.isApprox(Jinv.inverse()));
};
