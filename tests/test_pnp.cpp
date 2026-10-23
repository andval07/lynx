#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "lynx/geometry.hpp"

using namespace lynx;

namespace {

const Camera kCam{517.3, 516.5, 318.6, 255.3};

struct Scene {
  SE3 T_cw;
  std::vector<Vec3> pw;
  std::vector<Vec2> uv;
};

// Points spread in front of a camera with a random pose, seen in the image.
Scene make_scene(std::mt19937& rng, int n, double noise_px) {
  std::uniform_real_distribution<double> u(-1, 1), z(1.0, 5.0);
  std::normal_distribution<double> px(0.0, noise_px);
  Vec6 xi;
  for (int i = 0; i < 6; ++i) xi(i) = u(rng);
  Scene s;
  s.T_cw = SE3::exp(xi);
  const SE3 T_wc = s.T_cw.inverse();
  while (static_cast<int>(s.pw.size()) < n) {
    const Vec3 pc(u(rng) * 2, u(rng) * 1.5, z(rng));
    const Vec2 uv = kCam.project(pc);
    if (uv.x() < 0 || uv.x() > 640 || uv.y() < 0 || uv.y() > 480) continue;
    s.pw.push_back(T_wc * pc);
    s.uv.push_back(uv + Vec2(px(rng), px(rng)));
  }
  return s;
}

double rotation_error_deg(const SE3& a, const SE3& b) {
  return so3_log(a.R().transpose() * b.R()).norm() * 180.0 / M_PI;
}

}  // namespace

TEST(EPnP, ExactOnNoiseFreeData) {
  std::mt19937 rng(10);
  for (int trial = 0; trial < 20; ++trial) {
    const Scene s = make_scene(rng, 50, 0.0);
    const auto T = epnp(s.pw, s.uv, kCam);
    ASSERT_TRUE(T.has_value());
    EXPECT_LT(rotation_error_deg(*T, s.T_cw), 1e-6);
    EXPECT_LT((T->t() - s.T_cw.t()).norm(), 1e-6);
  }
}

TEST(EPnP, MinimalSixPoints) {
  std::mt19937 rng(11);
  const Scene s = make_scene(rng, 6, 0.0);
  const auto T = epnp(s.pw, s.uv, kCam);
  ASSERT_TRUE(T.has_value());
  EXPECT_LT((T->t() - s.T_cw.t()).norm(), 1e-5);
}

TEST(EPnP, NearlyPlanarScene) {
  // A desk: all points within 1 mm of a plane.
  std::mt19937 rng(12);
  Scene s = make_scene(rng, 80, 0.0);
  const SE3 T_wc = s.T_cw.inverse();
  std::uniform_real_distribution<double> u(-1, 1), e(-1e-3, 1e-3);
  for (size_t i = 0; i < s.pw.size(); ++i) {
    const Vec3 pc(u(rng) * 1.5, u(rng), 3.0 + e(rng));
    s.pw[i] = T_wc * pc;
    s.uv[i] = kCam.project(pc);
  }
  const auto T = epnp(s.pw, s.uv, kCam);
  ASSERT_TRUE(T.has_value());
  EXPECT_LT((T->t() - s.T_cw.t()).norm(), 1e-3);
}

TEST(EPnP, RejectsTooFewPoints) {
  std::mt19937 rng(13);
  const Scene s = make_scene(rng, 5, 0.0);
  EXPECT_FALSE(epnp(s.pw, s.uv, kCam).has_value());
}

TEST(RefinePose, ConvergesFromPerturbedStart) {
  std::mt19937 rng(14);
  const Scene s = make_scene(rng, 100, 0.0);
  Vec6 d;
  d << 0.05, -0.04, 0.03, 0.02, -0.03, 0.01;
  std::vector<int> all(s.pw.size());
  for (size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int>(i);
  const SE3 T = refine_pose(s.pw, s.uv, all, kCam, SE3::exp(d) * s.T_cw);
  EXPECT_LT((T.t() - s.T_cw.t()).norm(), 1e-8);
  EXPECT_LT(rotation_error_deg(T, s.T_cw), 1e-6);
}

TEST(PnPRansac, RecoversPoseWithFortyPercentOutliers) {
  std::mt19937 rng(15);
  Scene s = make_scene(rng, 200, 0.7);
  std::uniform_real_distribution<double> ux(0, 640), uy(0, 480);
  std::vector<bool> outlier(s.uv.size(), false);
  for (size_t i = 0; i < s.uv.size(); i += 5) {  // 2 of every 5
    s.uv[i] = Vec2(ux(rng), uy(rng));
    s.uv[i + 1] = Vec2(ux(rng), uy(rng));
    outlier[i] = outlier[i + 1] = true;
  }
  std::mt19937 rr(16);
  const PnPResult r = pnp_ransac(s.pw, s.uv, kCam, RansacParams{}, rr);
  ASSERT_TRUE(r.ok);
  EXPECT_LT((r.T_cw.t() - s.T_cw.t()).norm(), 0.02);
  EXPECT_LT(rotation_error_deg(r.T_cw, s.T_cw), 0.3);

  // Almost all inliers found, and almost no outlier accepted.
  int true_in = 0, false_in = 0;
  for (int i : r.inliers) (outlier[i] ? false_in : true_in)++;
  EXPECT_GE(true_in, 110);
  EXPECT_LE(false_in, 3);
}

TEST(PnPRansac, FailsOnPureNoise) {
  std::mt19937 rng(17);
  Scene s = make_scene(rng, 100, 0.0);
  std::uniform_real_distribution<double> ux(0, 640), uy(0, 480);
  for (auto& p : s.uv) p = Vec2(ux(rng), uy(rng));
  RansacParams params;
  params.min_inliers = 30;
  std::mt19937 rr(18);
  EXPECT_FALSE(pnp_ransac(s.pw, s.uv, kCam, params, rr).ok);
}

TEST(RansacIterations, LowInlierRatioHitsCapWithoutOverflow) {
  // 13 inliers out of 680: k ~ 1e11, far beyond INT_MAX.
  EXPECT_EQ(ransac_iterations(13.0 / 680, 6, 0.999, 300), 300);
  EXPECT_EQ(ransac_iterations(0.0, 6, 0.999, 300), 300);
}

TEST(RansacIterations, HighInlierRatioNeedsFewIterations) {
  EXPECT_EQ(ransac_iterations(1.0, 6, 0.999, 300), 1);
  // 0.9^6 = 0.531: log(0.001) / log(0.469) = 9.1
  EXPECT_EQ(ransac_iterations(0.9, 6, 0.999, 300), 10);
}
