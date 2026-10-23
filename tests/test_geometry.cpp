#include <gtest/gtest.h>

#include <random>

#include "lynx/geometry.hpp"

using namespace lynx;

namespace {

const Camera kCam{517.3, 516.5, 318.6, 255.3};

SE3 make_pose(double rx, double ry, double rz, double tx, double ty, double tz) {
  Vec6 xi;
  xi << tx, ty, tz, rx, ry, rz;
  return SE3::exp(xi);
}

}  // namespace

TEST(Triangulate, ExactWithoutNoise) {
  const SE3 T1;  // first camera at the origin
  const SE3 T2 = make_pose(0.02, -0.1, 0.03, -0.3, 0.05, 0.02);
  const Mat34 P1 = projection_matrix(kCam, T1);
  const Mat34 P2 = projection_matrix(kCam, T2);
  std::mt19937 rng(4);
  std::uniform_real_distribution<double> xy(-1.5, 1.5), z(1.0, 6.0);
  for (int i = 0; i < 50; ++i) {
    const Vec3 X(xy(rng), xy(rng), z(rng));
    const Vec3 Y = triangulate(P1, P2, kCam.project(T1 * X), kCam.project(T2 * X));
    EXPECT_LT((X - Y).norm(), 1e-8);
  }
}

TEST(Triangulate, PixelNoiseGivesSmallError) {
  const SE3 T2 = make_pose(0, -0.05, 0, -0.5, 0, 0);  // 50 cm baseline
  const Mat34 P1 = projection_matrix(kCam, SE3());
  const Mat34 P2 = projection_matrix(kCam, T2);
  std::mt19937 rng(5);
  std::normal_distribution<double> px(0.0, 0.5);
  const Vec3 X(0.2, -0.1, 3.0);
  double worst = 0;
  for (int i = 0; i < 100; ++i) {
    const Vec2 n1(px(rng), px(rng)), n2(px(rng), px(rng));
    const Vec3 Y = triangulate(P1, P2, kCam.project(X) + n1,
                               kCam.project(T2 * X) + n2);
    worst = std::max(worst, (X - Y).norm());
  }
  // Depth error ~ z^2 / (f b) * sigma = 9 / 258 * 0.5 ~ 1.7 cm per sigma.
  EXPECT_LT(worst, 0.1);
}

TEST(Camera, BackprojectInvertsProject) {
  const Vec3 p(0.4, -0.3, 2.5);
  const Vec3 q = kCam.backproject(kCam.project(p), p.z());
  EXPECT_LT((p - q).norm(), 1e-12);
}
