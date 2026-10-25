#pragma once

#include <vector>

#include "lynx/camera.hpp"
#include "lynx/se3.hpp"

namespace lynx {

// One measurement of point `point` in camera `pose`. For RGB-D points the
// depth enters as a virtual right-image coordinate u_r = u - bf / depth, as in
// a stereo rig with baseline b. u_right < 0 means no depth: 2D residual only.
struct BaObservation {
  int pose = 0;
  int point = 0;
  Vec2 uv;
  double u_right = -1;
  // Pixel noise of the measurement, e.g. the ORB pyramid scale 1.2^octave.
  // Residuals are divided by it, so Huber thresholds are in sigmas.
  double sigma = 1.0;
};

struct BaProblem {
  std::vector<SE3> T_cw;     // world-to-camera poses
  std::vector<bool> fixed;   // same size as T_cw, fixed poses are not moved
  std::vector<Vec3> points;  // world coordinates
  std::vector<BaObservation> obs;
};

struct BaOptions {
  double bf = 40.0;  // virtual baseline times fx, pixels * meters
  // Huber thresholds: square roots of the 95% chi-square values, 2 and 3 DoF.
  double huber_mono = 2.448;
  double huber_stereo = 2.796;
  int max_iters = 10;
};

struct BaResult {
  std::vector<SE3> T_cw;
  std::vector<Vec3> points;
  // Per observation: residual norm above its Huber threshold after the solve.
  std::vector<bool> outlier;
  double initial_cost = 0;
  double final_cost = 0;
  int iterations = 0;
};

// Levenberg-Marquardt on the reprojection error with a Huber kernel. The
// points are eliminated with the Schur complement, so the linear system has
// only the free poses. Left perturbation on T_cw, tangent order (rho, phi).
BaResult bundle_adjust(const BaProblem& problem, const Camera& cam,
                       const BaOptions& opt = {});

}  // namespace lynx
