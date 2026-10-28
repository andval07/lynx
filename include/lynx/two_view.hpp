#pragma once

#include <random>
#include <vector>

#include "lynx/camera.hpp"
#include "lynx/se3.hpp"

namespace lynx {

// Two-view geometry for the monocular start. Points are normalized camera
// coordinates (x / z, y / z) unless the name says pixels.

// Eight-point essential matrix, with the singular values forced to (1, 1, 0).
// Needs at least 8 pairs with x2^T E x1 = 0.
Mat3 essential_8point(const std::vector<Vec2>& x1, const std::vector<Vec2>& x2);

// DLT homography with x2 ~ H x1. Needs at least 4 pairs.
Mat3 homography_dlt(const std::vector<Vec2>& x1, const std::vector<Vec2>& x2);

// The four (R, t) of E = hat(t) R, with |t| = 1. Pose of camera 2 from camera 1.
std::vector<SE3> decompose_essential(const Mat3& E);

// The (R, t / d) of H = R + t n^T / d (Ma et al., "An Invitation to 3-D
// Vision", 5.3). Empty when H is close to a pure rotation.
std::vector<SE3> decompose_homography(const Mat3& H);

struct TwoViewParams {
  double sigma_px = 1.0;          // keypoint noise used to score the models
  int max_iters = 200;            // RANSAC iterations, shared by E and H
  double min_parallax_deg = 1.0;  // median parallax of the triangulated points
  int min_points = 50;            // triangulated points needed
  double homography_ratio = 0.45; // pick H when S_H / (S_H + S_E) is above
};

struct TwoViewResult {
  bool ok = false;
  bool homography = false;      // which model won
  SE3 T_21;                     // camera 1 to camera 2, |t| = 1
  std::vector<Vec3> points;     // in camera 1, valid where good[i]
  std::vector<bool> good;
  double parallax_deg = 0;
};

// Monocular initialization from matched pixels: E and H in RANSAC, model
// choice by score as in ORB-SLAM, then the decomposition that puts the most
// points in front of both cameras. The scale is the one of |t| = 1.
TwoViewResult initialize_two_view(const std::vector<Vec2>& uv1,
                                  const std::vector<Vec2>& uv2,
                                  const Camera& cam, const TwoViewParams& p,
                                  std::mt19937& rng);

}  // namespace lynx
