#include <gtest/gtest.h>

#include <random>

#include "lynx/ba.hpp"

using namespace lynx;

namespace {

const Camera kCam{517.3, 516.5, 318.6, 255.3};
constexpr double kBf = 40.0;

// Cameras on a short arc looking at a box of points 1.5-4 m away.
struct Scene {
  std::vector<SE3> T_cw;
  std::vector<Vec3> points;
  BaProblem problem;
};

Scene make_scene(std::mt19937& rng, int n_poses, int n_points, double noise_px,
                 bool stereo) {
  std::uniform_real_distribution<double> u(-1, 1), z(1.5, 4.0);
  std::normal_distribution<double> px(0.0, noise_px);
  Scene s;
  for (int i = 0; i < n_poses; ++i) {
    Vec6 xi;
    xi << 0.1 * i, 0.02 * i, 0.03 * i, 0.01 * i, -0.04 * i, 0.02 * i;
    s.T_cw.push_back(SE3::exp(xi));
  }
  for (int j = 0; j < n_points; ++j) {
    s.points.emplace_back(u(rng) * 1.5, u(rng) * 1.0, z(rng));
  }
  s.problem.T_cw = s.T_cw;
  s.problem.fixed.assign(n_poses, false);
  s.problem.points = s.points;
  for (int i = 0; i < n_poses; ++i) {
    for (int j = 0; j < n_points; ++j) {
      const Vec3 pc = s.T_cw[i] * s.points[j];
      if (pc.z() <= 0.1) continue;
      const Vec2 uv = kCam.project(pc);
      if (uv.x() < 0 || uv.x() > 640 || uv.y() < 0 || uv.y() > 480) continue;
      BaObservation o;
      o.pose = i;
      o.point = j;
      o.uv = uv + Vec2(px(rng), px(rng));
      if (stereo) o.u_right = uv.x() - kBf / pc.z() + px(rng);
      s.problem.obs.push_back(o);
    }
  }
  return s;
}

void perturb(BaProblem& p, std::mt19937& rng, double pose_sigma,
             double point_sigma) {
  std::normal_distribution<double> n(0.0, 1.0);
  for (size_t i = 0; i < p.T_cw.size(); ++i) {
    if (p.fixed[i]) continue;
    Vec6 xi;
    for (int k = 0; k < 6; ++k) xi(k) = pose_sigma * n(rng);
    p.T_cw[i] = SE3::exp(xi) * p.T_cw[i];
  }
  for (auto& pt : p.points) pt += point_sigma * Vec3(n(rng), n(rng), n(rng));
}

double max_position_error(const std::vector<SE3>& a, const std::vector<SE3>& b) {
  double e = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    e = std::max(e, (a[i].inverse().t() - b[i].inverse().t()).norm());
  }
  return e;
}

BaOptions options() {
  BaOptions opt;
  opt.bf = kBf;
  opt.max_iters = 20;
  return opt;
}

}  // namespace

TEST(BundleAdjustment, ExactDataStaysPut) {
  std::mt19937 rng(1);
  Scene s = make_scene(rng, 4, 100, 0.0, true);
  s.problem.fixed[0] = true;
  const BaResult r = bundle_adjust(s.problem, kCam, options());
  EXPECT_LT(r.final_cost, 1e-12);
  EXPECT_LT(max_position_error(r.T_cw, s.T_cw), 1e-9);
}

TEST(BundleAdjustment, RgbdRecoversPosesAndPoints) {
  std::mt19937 rng(2);
  Scene s = make_scene(rng, 6, 300, 0.5, true);
  s.problem.fixed[0] = true;
  perturb(s.problem, rng, 0.02, 0.05);
  const double before = max_position_error(s.problem.T_cw, s.T_cw);

  const BaResult r = bundle_adjust(s.problem, kCam, options());

  EXPECT_LT(r.final_cost, 0.05 * r.initial_cost);
  EXPECT_LT(max_position_error(r.T_cw, s.T_cw), 0.005) << "before " << before;
  // Depth from a 40 px*m virtual baseline is coarse far away (about 20 cm at
  // 4 m), so check the points on average against the perturbed start.
  double before_sum = 0, after_sum = 0;
  for (size_t j = 0; j < s.points.size(); ++j) {
    before_sum += (s.problem.points[j] - s.points[j]).norm();
    after_sum += (r.points[j] - s.points[j]).norm();
  }
  EXPECT_LT(after_sum, 0.5 * before_sum);
}

TEST(BundleAdjustment, MonoWithTwoFixedPoses) {
  // Two fixed cameras set the scale, depth is not needed.
  std::mt19937 rng(3);
  Scene s = make_scene(rng, 5, 300, 0.5, false);
  s.problem.fixed[0] = s.problem.fixed[1] = true;
  perturb(s.problem, rng, 0.01, 0.03);

  const BaResult r = bundle_adjust(s.problem, kCam, options());

  EXPECT_LT(r.final_cost, 0.05 * r.initial_cost);
  EXPECT_LT(max_position_error(r.T_cw, s.T_cw), 0.01);
}

TEST(BundleAdjustment, FixedPosesDoNotMove) {
  std::mt19937 rng(4);
  Scene s = make_scene(rng, 4, 150, 0.5, true);
  s.problem.fixed[0] = s.problem.fixed[2] = true;
  perturb(s.problem, rng, 0.0, 0.05);
  const BaResult r = bundle_adjust(s.problem, kCam, options());
  for (int i : {0, 2}) {
    EXPECT_TRUE(r.T_cw[i].matrix().isApprox(s.problem.T_cw[i].matrix(), 0.0));
  }
}

TEST(BundleAdjustment, HuberKeepsOutliersFromPullingPoses) {
  std::mt19937 rng(5);
  Scene s = make_scene(rng, 5, 300, 0.5, true);
  s.problem.fixed[0] = true;
  std::vector<bool> corrupted(s.problem.obs.size(), false);
  for (size_t k = 0; k < s.problem.obs.size(); k += 10) {
    s.problem.obs[k].uv += Vec2(30, -25);
    corrupted[k] = true;
  }
  perturb(s.problem, rng, 0.01, 0.03);

  const BaResult r = bundle_adjust(s.problem, kCam, options());

  EXPECT_LT(max_position_error(r.T_cw, s.T_cw), 0.01);
  int caught = 0, false_alarms = 0, n_bad = 0;
  for (size_t k = 0; k < corrupted.size(); ++k) {
    n_bad += corrupted[k];
    if (r.outlier[k] && corrupted[k]) ++caught;
    if (r.outlier[k] && !corrupted[k]) ++false_alarms;
  }
  EXPECT_EQ(caught, n_bad);
  EXPECT_LT(false_alarms, static_cast<int>(0.1 * corrupted.size()));
}
