#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>

#include "lynx/trajectory.hpp"
#include "lynx/tum.hpp"

using namespace lynx;

namespace {

// A helix, so that the alignment is fully constrained.
Trajectory helix(int n, double t0) {
  Trajectory t;
  for (int i = 0; i < n; ++i) {
    const double a = 0.1 * i;
    t.push_back({t0 + 0.033 * i, SE3(Mat3::Identity(),
                                     Vec3(std::cos(a), std::sin(a), 0.05 * i))});
  }
  return t;
}

}  // namespace

TEST(Associate, PairsNearestWithinTolerance) {
  Trajectory a = helix(5, 0.0);
  Trajectory b = helix(5, 0.005);
  b.erase(b.begin() + 2);
  const auto pairs = associate(a, b, 0.02);
  ASSERT_EQ(pairs.size(), 4u);
  EXPECT_EQ(pairs[2], std::make_pair(3, 2));
}

TEST(Associate, IgnoresFarStamps) {
  EXPECT_TRUE(associate(helix(5, 0.0), helix(5, 10.0)).empty());
}

TEST(Umeyama, RecoversRigidTransform) {
  Vec6 xi;
  xi << 0.5, -1, 2, 0.3, -0.2, 1.0;
  const SE3 T = SE3::exp(xi);
  std::vector<Vec3> src, dst;
  for (const auto& p : helix(30, 0)) {
    src.push_back(p.T_wc.t());
    dst.push_back(T * p.T_wc.t());
  }
  EXPECT_TRUE(align_umeyama(src, dst).matrix().isApprox(T.matrix(), 1e-10));
}

TEST(Ate, ZeroForRigidlyMovedCopy) {
  Vec6 xi;
  xi << 1, 2, 3, 0.1, 0.2, 0.3;
  const SE3 T = SE3::exp(xi);
  const Trajectory gt = helix(40, 0);
  Trajectory est;
  for (const auto& p : gt) est.push_back({p.stamp, T * p.T_wc});
  EXPECT_LT(ate_rmse(est, gt), 1e-9);
}

TEST(Ate, MeasuresConstantOffsetAlongOneAxis) {
  // Shift every other pose by +/- 1 cm in z: no rigid motion can remove it.
  const Trajectory gt = helix(40, 0);
  Trajectory est = gt;
  for (size_t i = 0; i < est.size(); ++i) {
    const double dz = i % 2 ? 0.01 : -0.01;
    est[i].T_wc = SE3(Mat3::Identity(), est[i].T_wc.t() + Vec3(0, 0, dz));
  }
  EXPECT_NEAR(ate_rmse(est, gt), 0.01, 1e-3);
}

TEST(TumIo, WriteThenReadKeepsPoses) {
  Trajectory t = helix(10, 1305031102.175304);
  Vec6 xi;
  xi << 0, 0, 0, 0.3, -0.5, 0.2;
  for (auto& p : t) p.T_wc = p.T_wc * SE3::exp(xi);
  const auto path = std::filesystem::temp_directory_path() / "lynx_traj.txt";
  write_tum_trajectory(path.string(), t);
  const Trajectory back = read_tum_trajectory(path.string());
  ASSERT_EQ(back.size(), t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(back[i].stamp, t[i].stamp, 1e-6);
    EXPECT_TRUE(back[i].T_wc.matrix().isApprox(t[i].T_wc.matrix(), 1e-7));
  }
  std::filesystem::remove(path);
}

TEST(TumIo, CalibrationFromName) {
  EXPECT_DOUBLE_EQ(tum_calibration("x/rgbd_dataset_freiburg1_xyz").cam.fx, 517.3);
  EXPECT_DOUBLE_EQ(tum_calibration("rgbd_dataset_freiburg3_long").dist[0], 0.0);
  EXPECT_THROW(tum_calibration("kitti_00"), std::runtime_error);
}
