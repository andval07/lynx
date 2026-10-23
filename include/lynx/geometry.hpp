#pragma once

#include <optional>
#include <random>
#include <vector>

#include "lynx/camera.hpp"
#include "lynx/se3.hpp"

namespace lynx {

using Mat34 = Eigen::Matrix<double, 3, 4>;

// Linear (DLT) triangulation of one point seen by two projection matrices.
// x1, x2 are in the same coordinates the matrices map to (pixels or normalized).
Vec3 triangulate(const Mat34& P1, const Mat34& P2, const Vec2& x1,
                 const Vec2& x2);

// Projection matrix K [R | t] for a world-to-camera pose.
Mat34 projection_matrix(const Camera& cam, const SE3& T_cw);

// EPnP (Lepetit et al. 2009). Needs at least 6 points.
// Returns the world-to-camera pose, or nothing if the input is degenerate.
std::optional<SE3> epnp(const std::vector<Vec3>& pw, const std::vector<Vec2>& uv,
                        const Camera& cam);

// Levenberg-Marquardt on the reprojection error with a Huber kernel.
// Only the points listed in idx are used. Returns the refined T_cw.
SE3 refine_pose(const std::vector<Vec3>& pw, const std::vector<Vec2>& uv,
                const std::vector<int>& idx, const Camera& cam, const SE3& T_cw,
                double huber_px = 2.0, int max_iters = 20);

struct RansacParams {
  double threshold_px = 3.0;
  int max_iters = 300;
  double confidence = 0.999;
  int min_inliers = 15;
};

struct PnPResult {
  bool ok = false;
  SE3 T_cw;
  std::vector<int> inliers;
};

// Iterations needed to draw one all-inlier sample with the given confidence,
// capped at `cap`.
int ransac_iterations(double inlier_ratio, int sample_size, double confidence,
                      int cap);

// EPnP inside RANSAC, then LM refinement on the inliers.
PnPResult pnp_ransac(const std::vector<Vec3>& pw, const std::vector<Vec2>& uv,
                     const Camera& cam, const RansacParams& params,
                     std::mt19937& rng);

// Indices of the points with reprojection error below threshold_px.
std::vector<int> find_inliers(const std::vector<Vec3>& pw,
                              const std::vector<Vec2>& uv, const Camera& cam,
                              const SE3& T_cw, double threshold_px);

}  // namespace lynx
