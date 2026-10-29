#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "lynx/mono_odometry.hpp"

using namespace lynx;

namespace {

const TumCalibration kCalib{{517.3, 516.5, 318.6, 255.3}, {}, 5000.0};

struct World {
  std::vector<Vec3> points;
  cv::Mat descriptors;  // one random ORB-like row per point
};

World make_world(int n, std::mt19937& rng) {
  std::uniform_real_distribution<double> x(-4.0, 5.0), y(-2.0, 2.0), z(3.0, 7.0);
  std::uniform_int_distribution<int> byte(0, 255);
  World w;
  w.descriptors = cv::Mat(n, 32, CV_8U);
  for (int i = 0; i < n; ++i) {
    w.points.emplace_back(x(rng), y(rng), z(rng));
    for (int b = 0; b < 32; ++b) w.descriptors.at<uint8_t>(i, b) = static_cast<uint8_t>(byte(rng));
  }
  return w;
}

// Camera moving sideways with a slow yaw; pose of frame k in the world.
SE3 gt_pose(int k) {
  Vec6 xi;
  xi << 0.03 * k, 0.002 * k, 0.0, 0.0, -0.002 * k, 0.0;
  return SE3::exp(xi);
}

Features observe(const World& w, const SE3& T_wc, std::mt19937& rng) {
  std::normal_distribution<double> px(0.0, 0.5);
  const SE3 T_cw = T_wc.inverse();
  Features f;
  std::vector<int> rows;
  for (size_t i = 0; i < w.points.size(); ++i) {
    const Vec3 pc = T_cw * w.points[i];
    if (pc.z() < 0.5) continue;
    const Vec2 uv = kCalib.cam.project(pc) + Vec2(px(rng), px(rng));
    if (uv.x() < 0 || uv.y() < 0 || uv.x() >= 640 || uv.y() >= 480) continue;
    f.uv.push_back(uv);
    f.keypoints.emplace_back(cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y())), 31.0f, -1.0f, 0.0f, 0);
    rows.push_back(static_cast<int>(i));
  }
  f.descriptors = cv::Mat(static_cast<int>(rows.size()), 32, CV_8U);
  for (size_t j = 0; j < rows.size(); ++j) {
    w.descriptors.row(rows[j]).copyTo(f.descriptors.row(static_cast<int>(j)));
  }
  return f;
}

double rot_err_deg(const Mat3& A, const Mat3& B) {
  return so3_log(A.transpose() * B).norm() * 180.0 / M_PI;
}

}  // namespace

TEST(MonoOdometry, TracksSyntheticSequenceUpToScale) {
  std::mt19937 rng(7);
  const World w = make_world(800, rng);
  MonoOdometry vo(kCalib);
  const int n = 60;
  int first = -1;
  for (int k = 0; k < n; ++k) {
    const FrameResult r = vo.track(observe(w, gt_pose(k), rng));
    if (first < 0 && r.tracked) first = k;
    if (first >= 0) EXPECT_TRUE(r.tracked) << "frame " << k;
  }
  ASSERT_GE(first, 1);
  EXPECT_LE(first, 10);  // enough parallax after a few frames
  EXPECT_GE(vo.keyframes(), 4);

  // Frame 0 is the reference of the initialization, so it is the map origin.
  const auto est = vo.refined_poses();
  const int ref = 0;
  // One least-squares scale for the whole run, as with a Sim(3) alignment.
  double num = 0, den = 0;
  for (int k = first; k < n; ++k) {
    const Vec3 g = (gt_pose(ref).inverse() * gt_pose(k)).t();
    const Vec3 e = (est[ref].inverse() * est[k]).t();
    num += g.dot(e);
    den += e.dot(e);
  }
  const double scale = num / den;
  for (int k = first; k < n; ++k) {
    const SE3 gt_rel = gt_pose(ref).inverse() * gt_pose(k);
    const SE3 es_rel = est[ref].inverse() * est[k];
    EXPECT_LT(rot_err_deg(es_rel.R(), gt_rel.R()), 0.5) << "frame " << k;
    EXPECT_LT((es_rel.t() * scale - gt_rel.t()).norm(), 0.05 * gt_rel.t().norm() + 0.01)
        << "frame " << k;
  }
}
