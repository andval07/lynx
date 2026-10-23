#include <gtest/gtest.h>

#include <random>

#include "lynx/se3.hpp"

using namespace lynx;

namespace {

Vec6 random_twist(std::mt19937& rng, double scale) {
  std::uniform_real_distribution<double> d(-scale, scale);
  Vec6 xi;
  for (int i = 0; i < 6; ++i) xi(i) = d(rng);
  return xi;
}

}  // namespace

TEST(SO3, ExpOfZeroIsIdentity) {
  EXPECT_TRUE(so3_exp(Vec3::Zero()).isApprox(Mat3::Identity()));
}

TEST(SO3, ExpMatchesEigenAngleAxis) {
  const Vec3 w(0.3, -1.2, 0.7);
  const Mat3 ref = Eigen::AngleAxisd(w.norm(), w.normalized()).toRotationMatrix();
  EXPECT_TRUE(so3_exp(w).isApprox(ref, 1e-12));
}

TEST(SO3, LogInvertsExpAcrossAngles) {
  // Small, generic and almost-pi rotations all take different branches.
  for (double angle : {1e-9, 1e-4, 0.5, 2.0, 3.0, M_PI - 1e-5, M_PI}) {
    const Vec3 w = angle * Vec3(1, 2, -0.5).normalized();
    const Vec3 back = so3_log(so3_exp(w));
    EXPECT_TRUE(so3_exp(back).isApprox(so3_exp(w), 1e-9)) << "angle " << angle;
    if (angle < M_PI - 1e-3) {
      EXPECT_LT((back - w).norm(), 1e-9) << "angle " << angle;
    }
  }
}

TEST(SO3, ExpIsOrthonormal) {
  const Mat3 R = so3_exp(Vec3(2.0, -0.4, 1.1));
  EXPECT_TRUE((R.transpose() * R).isApprox(Mat3::Identity(), 1e-12));
  EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
}

TEST(SO3, LeftJacobianMatchesFiniteDifference) {
  // exp(phi + d) ~ exp(J(phi) d) exp(phi) for small d.
  const Vec3 phi(0.4, -0.9, 0.3);
  const Mat3 J = so3_left_jacobian(phi);
  const double h = 1e-6;
  for (int k = 0; k < 3; ++k) {
    const Vec3 d = h * Vec3::Unit(k);
    const Vec3 col = so3_log(so3_exp(phi + d) * so3_exp(phi).transpose()) / h;
    EXPECT_LT((col - J.col(k)).norm(), 1e-5) << "column " << k;
  }
}

TEST(SO3, LeftJacobianInverse) {
  for (const Vec3& phi : {Vec3(1e-8, 0, 0), Vec3(0.4, -0.9, 0.3), Vec3(2, 1, -1)}) {
    const Mat3 P = so3_left_jacobian(phi) * so3_left_jacobian_inv(phi);
    EXPECT_TRUE(P.isApprox(Mat3::Identity(), 1e-10));
  }
}

TEST(SE3, ExpLogRoundTrip) {
  std::mt19937 rng(1);
  for (int i = 0; i < 100; ++i) {
    const Vec6 xi = random_twist(rng, 1.5);
    EXPECT_LT((SE3::exp(xi).log() - xi).norm(), 1e-9);
  }
}

TEST(SE3, InverseAndComposition) {
  std::mt19937 rng(2);
  const SE3 A = SE3::exp(random_twist(rng, 1.0));
  const SE3 B = SE3::exp(random_twist(rng, 1.0));
  EXPECT_TRUE((A * A.inverse()).matrix().isApprox(Eigen::Matrix4d::Identity(), 1e-12));
  EXPECT_TRUE((A * B).matrix().isApprox(A.matrix() * B.matrix(), 1e-12));
  const Vec3 p(0.3, -2, 5);
  EXPECT_TRUE(((A * B) * p).isApprox(A * (B * p), 1e-12));
}

TEST(SE3, ExpOfPureTranslation) {
  Vec6 xi;
  xi << 1, 2, 3, 0, 0, 0;
  const SE3 T = SE3::exp(xi);
  EXPECT_TRUE(T.R().isApprox(Mat3::Identity()));
  EXPECT_TRUE(T.t().isApprox(Vec3(1, 2, 3)));
}

TEST(SE3, AdjointMovesTwistsAcrossFrames) {
  std::mt19937 rng(3);
  const SE3 T = SE3::exp(random_twist(rng, 1.0));
  const Vec6 xi = random_twist(rng, 0.5);
  const SE3 lhs = T * SE3::exp(xi) * T.inverse();
  const SE3 rhs = SE3::exp(T.adjoint() * xi);
  EXPECT_TRUE(lhs.matrix().isApprox(rhs.matrix(), 1e-10));
}
