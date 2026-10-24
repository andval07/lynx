// End-to-end: the first frames of TUM fr1/xyz. Skipped when the dataset has
// not been downloaded into data/.

#include <gtest/gtest.h>

#include <filesystem>

#include "lynx/odometry.hpp"

using namespace lynx;

TEST(EndToEnd, Freiburg1XyzFirst150Frames) {
  const std::string dir = std::string(LYNX_DATA_DIR) + "/rgbd_dataset_freiburg1_xyz";
  if (!std::filesystem::exists(dir + "/rgb.txt")) {
    GTEST_SKIP() << "dataset not found in " << dir;
  }
  const RunStats s = run_tum_sequence(dir, 150);
  ASSERT_EQ(s.frames, 150);
  EXPECT_EQ(s.lost, 0);
  const double ate = ate_rmse(s.trajectory, read_tum_trajectory(dir + "/groundtruth.txt"));
  ASSERT_GT(ate, 0.0);
  EXPECT_LT(ate, 0.03) << "ATE " << ate << " m";
}
