#pragma once

#include <utility>
#include <vector>

#include "lynx/se3.hpp"

namespace lynx {

struct StampedPose {
  double stamp = 0;
  SE3 T_wc;  // camera-to-world
};
using Trajectory = std::vector<StampedPose>;

// Pairs (i, j) with |a[i].stamp - b[j].stamp| <= max_dt, nearest match first,
// each index used once. Both inputs must be sorted by stamp.
std::vector<std::pair<int, int>> associate(const Trajectory& a,
                                           const Trajectory& b,
                                           double max_dt = 0.02);

// Rigid (no scale) Umeyama alignment: T such that dst ~ T * src.
SE3 align_umeyama(const std::vector<Vec3>& src, const std::vector<Vec3>& dst);

// Absolute trajectory error: RMSE of the positions after rigid alignment of
// the estimate onto the ground truth. Returns -1 if fewer than 3 matches.
double ate_rmse(const Trajectory& estimate, const Trajectory& ground_truth);

}  // namespace lynx
