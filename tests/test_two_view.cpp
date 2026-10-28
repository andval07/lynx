#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "lynx/two_view.hpp"

using namespace lynx;

namespace {

const Camera kCam{517.3, 516.5, 318.6, 255.3};

SE3 make_pose(double rx, double ry, double rz, double tx, double ty, double tz) {
  Vec6 xi;
  xi << tx, ty, tz, rx, ry, rz;
  return SE3::exp(xi);
}

Vec2 norm_proj(const Vec3& p) { return p.head<2>() / p.z(); }

double rot_err_deg(const Mat3& A, const Mat3& B) {
  return so3_log(A.transpose() * B).norm() * 180.0 / M_PI;
}

double dir_err_deg(const Vec3& a, const Vec3& b) {
  const double c = a.normalized().dot(b.normalized());
  return std::acos(std::min(1.0, c)) * 180.0 / M_PI;
}

bool contains_pose(const std::vector<SE3>& cands, const SE3& T) {
  for (const SE3& c : cands) {
    if (rot_err_deg(c.R(), T.R()) < 1e-3 && dir_err_deg(c.t(), T.t()) < 1e-3) {
      return true;
    }
  }
  return false;
}

// Points in front of camera 1; flat = all on the plane z = 3.
std::vector<Vec3> make_scene(int n, bool flat, std::mt19937& rng) {
  std::uniform_real_distribution<double> xy(-1.5, 1.5), z(2.0, 6.0);
  std::vector<Vec3> pts;
  for (int i = 0; i < n; ++i) pts.emplace_back(xy(rng), xy(rng), flat ? 3.0 : z(rng));
  return pts;
}

}  // namespace

TEST(TwoView, EightPointEssentialIsExact) {
  std::mt19937 rng(1);
  const SE3 T21 = make_pose(0.05, -0.1, 0.02, -0.4, 0.1, 0.05);
  std::vector<Vec2> x1, x2;
  for (const Vec3& p : make_scene(30, false, rng)) {
    x1.push_back(norm_proj(p));
    x2.push_back(norm_proj(T21 * p));
  }
  const Mat3 E = essential_8point(x1, x2);
  for (size_t i = 0; i < x1.size(); ++i) {
    EXPECT_NEAR(x2[i].homogeneous().dot(E * x1[i].homogeneous()), 0.0, 1e-9);
  }
  Eigen::JacobiSVD<Mat3> svd(E);
  EXPECT_NEAR(svd.singularValues()(0), svd.singularValues()(1), 1e-9);
  EXPECT_NEAR(svd.singularValues()(2), 0.0, 1e-9);
  EXPECT_TRUE(contains_pose(decompose_essential(E), T21));
}

TEST(TwoView, HomographyDecompositionContainsTruth) {
  std::mt19937 rng(2);
  const SE3 T21 = make_pose(0.03, 0.08, -0.02, 0.3, -0.1, 0.1);
  std::vector<Vec2> x1, x2;
  for (const Vec3& p : make_scene(20, true, rng)) {
    x1.push_back(norm_proj(p));
    x2.push_back(norm_proj(T21 * p));
  }
  const Mat3 H = homography_dlt(x1, x2);
  for (size_t i = 0; i < x1.size(); ++i) {
    EXPECT_LT((norm_proj(H * x1[i].homogeneous()) - x2[i]).norm(), 1e-9);
  }
  EXPECT_TRUE(contains_pose(decompose_homography(H), T21));
}

TEST(TwoView, InitializesGeneralSceneWithNoiseAndOutliers) {
  std::mt19937 rng(3);
  const SE3 T21 = make_pose(0.02, -0.06, 0.01, -0.3, 0.02, 0.05);
  std::normal_distribution<double> px(0.0, 0.7);
  std::uniform_real_distribution<double> any_u(0, 640), any_v(0, 480);
  const auto scene = make_scene(300, false, rng);
  std::vector<Vec2> uv1, uv2;
  for (size_t i = 0; i < scene.size(); ++i) {
    uv1.push_back(kCam.project(scene[i]) + Vec2(px(rng), px(rng)));
    uv2.push_back(i % 5 == 0 ? Vec2(any_u(rng), any_v(rng))  // 20% outliers
                             : kCam.project(T21 * scene[i]) + Vec2(px(rng), px(rng)));
  }
  const TwoViewResult r = initialize_two_view(uv1, uv2, kCam, {}, rng);
  ASSERT_TRUE(r.ok);
  EXPECT_FALSE(r.homography);
  EXPECT_LT(rot_err_deg(r.T_21.R(), T21.R()), 0.3);
  EXPECT_LT(dir_err_deg(r.T_21.t(), T21.t()), 2.0);
  // Points come back up to the scale of |t| = 1.
  const double s = T21.t().norm();
  int good = 0, bad = 0;
  std::vector<double> rel;  // relative error of each good point
  for (size_t i = 0; i < scene.size(); ++i) {
    if (!r.good[i]) continue;
    // A random outlier can fall on its epipolar line by chance.
    if (i % 5 == 0) {
      ++bad;
      continue;
    }
    rel.push_back((r.points[i] * s - scene[i]).norm() / scene[i].norm());
    ++good;
  }
  EXPECT_GT(good, 200);
  EXPECT_LE(bad, 3);
  // Far points have a large depth error at this baseline, so check the bulk.
  std::sort(rel.begin(), rel.end());
  EXPECT_LT(rel[rel.size() / 2], 0.05);
  EXPECT_LT(rel.back(), 0.25);
}

TEST(TwoView, InitializesPlanarSceneWithHomography) {
  std::mt19937 rng(4);
  const SE3 T21 = make_pose(0.01, 0.05, 0.0, 0.3, 0.0, 0.05);
  std::normal_distribution<double> px(0.0, 0.5);
  std::vector<Vec2> uv1, uv2;
  for (const Vec3& p : make_scene(200, true, rng)) {
    uv1.push_back(kCam.project(p) + Vec2(px(rng), px(rng)));
    uv2.push_back(kCam.project(T21 * p) + Vec2(px(rng), px(rng)));
  }
  const TwoViewResult r = initialize_two_view(uv1, uv2, kCam, {}, rng);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(r.homography);
  EXPECT_LT(rot_err_deg(r.T_21.R(), T21.R()), 0.5);
  EXPECT_LT(dir_err_deg(r.T_21.t(), T21.t()), 3.0);
}

TEST(TwoView, RejectsPureRotation) {
  std::mt19937 rng(5);
  const SE3 T21 = make_pose(0.02, 0.06, 0.01, 0, 0, 0);
  std::normal_distribution<double> px(0.0, 0.5);
  std::vector<Vec2> uv1, uv2;
  for (const Vec3& p : make_scene(200, false, rng)) {
    uv1.push_back(kCam.project(p) + Vec2(px(rng), px(rng)));
    uv2.push_back(kCam.project(T21 * p) + Vec2(px(rng), px(rng)));
  }
  EXPECT_FALSE(initialize_two_view(uv1, uv2, kCam, {}, rng).ok);
}
