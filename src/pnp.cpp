#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "lynx/geometry.hpp"

namespace lynx {

namespace {

constexpr int kSampleSize = 6;

double huber_weight(double r, double k) { return r <= k ? 1.0 : k / r; }

// Cost of one pose: sum of Huber-weighted squared pixel errors.
double pose_cost(const std::vector<Vec3>& pw, const std::vector<Vec2>& uv,
                 const std::vector<int>& idx, const Camera& cam, const SE3& T,
                 double huber) {
  double cost = 0;
  for (int i : idx) {
    const Vec3 p = T * pw[i];
    if (p.z() <= 0) continue;
    const double r = (cam.project(p) - uv[i]).norm();
    cost += r <= huber ? r * r : 2 * huber * r - huber * huber;
  }
  return cost;
}

}  // namespace

int ransac_iterations(double inlier_ratio, int sample_size, double confidence,
                      int cap) {
  const double p_good = std::pow(inlier_ratio, sample_size);
  if (p_good >= 1.0) return 1;
  // log1p keeps precision when p_good is tiny. Clamp in double: k can be far
  // beyond INT_MAX for a low inlier ratio.
  const double k = std::log(1.0 - confidence) / std::log1p(-p_good);
  if (!(k < cap)) return cap;
  return std::max(1, static_cast<int>(std::ceil(k)));
}

SE3 refine_pose(const std::vector<Vec3>& pw, const std::vector<Vec2>& uv,
                const std::vector<int>& idx, const Camera& cam, const SE3& T_cw,
                double huber_px, int max_iters) {
  SE3 T = T_cw;
  double cost = pose_cost(pw, uv, idx, cam, T, huber_px);
  double lambda = 1e-3;
  for (int it = 0; it < max_iters; ++it) {
    // Left perturbation T <- exp(d) T, so d(T p)/dd = [I, -hat(T p)].
    Mat6 H = Mat6::Zero();
    Vec6 g = Vec6::Zero();
    for (int i : idx) {
      const Vec3 p = T * pw[i];
      if (p.z() <= 0) continue;
      const Vec2 e = cam.project(p) - uv[i];
      const double iz = 1.0 / p.z();
      Eigen::Matrix<double, 2, 3> Jp;
      Jp << cam.fx * iz, 0, -cam.fx * p.x() * iz * iz,
            0, cam.fy * iz, -cam.fy * p.y() * iz * iz;
      Eigen::Matrix<double, 3, 6> Jt;
      Jt << Mat3::Identity(), -hat(p);
      const Eigen::Matrix<double, 2, 6> J = Jp * Jt;
      const double w = huber_weight(e.norm(), huber_px);
      H += w * J.transpose() * J;
      g += w * J.transpose() * e;
    }
    bool improved = false;
    while (!improved && lambda < 1e8) {
      Mat6 A = H;
      A.diagonal() += lambda * H.diagonal();
      const Vec6 delta = A.ldlt().solve(-g);
      const SE3 T_new = SE3::exp(delta) * T;
      const double c = pose_cost(pw, uv, idx, cam, T_new, huber_px);
      if (c < cost) {
        T = T_new;
        cost = c;
        lambda = std::max(lambda / 10, 1e-9);
        improved = true;
        if (delta.norm() < 1e-10) return T;
      } else {
        lambda *= 10;
      }
    }
    if (!improved) break;
  }
  return T;
}

std::vector<int> find_inliers(const std::vector<Vec3>& pw,
                              const std::vector<Vec2>& uv, const Camera& cam,
                              const SE3& T_cw, double threshold_px) {
  std::vector<int> in;
  const double th2 = threshold_px * threshold_px;
  for (size_t i = 0; i < pw.size(); ++i) {
    const Vec3 p = T_cw * pw[i];
    if (p.z() <= 0) continue;
    if ((cam.project(p) - uv[i]).squaredNorm() < th2) {
      in.push_back(static_cast<int>(i));
    }
  }
  return in;
}

PnPResult pnp_ransac(const std::vector<Vec3>& pw, const std::vector<Vec2>& uv,
                     const Camera& cam, const RansacParams& params,
                     std::mt19937& rng) {
  PnPResult res;
  const int n = static_cast<int>(pw.size());
  if (n < std::max(kSampleSize, params.min_inliers)) return res;

  std::vector<int> all(n);
  std::iota(all.begin(), all.end(), 0);
  std::vector<Vec3> spw(kSampleSize);
  std::vector<Vec2> suv(kSampleSize);
  std::vector<int> best_in;
  SE3 best_T;
  int iters = params.max_iters;
  for (int it = 0; it < iters; ++it) {
    // Partial Fisher-Yates: the first kSampleSize entries are the sample.
    for (int k = 0; k < kSampleSize; ++k) {
      std::uniform_int_distribution<int> d(k, n - 1);
      std::swap(all[k], all[d(rng)]);
      spw[k] = pw[all[k]];
      suv[k] = uv[all[k]];
    }
    const auto T = epnp(spw, suv, cam);
    if (!T) continue;
    auto in = find_inliers(pw, uv, cam, *T, params.threshold_px);
    if (in.size() > best_in.size()) {
      best_in = std::move(in);
      best_T = *T;
      iters = ransac_iterations(static_cast<double>(best_in.size()) / n, kSampleSize,
                                params.confidence, params.max_iters);
    }
  }
  if (static_cast<int>(best_in.size()) < params.min_inliers) return res;

  // Refine on the inliers, then refine again on the inliers of the new pose.
  SE3 T = refine_pose(pw, uv, best_in, cam, best_T);
  for (int round = 0; round < 2; ++round) {
    auto in = find_inliers(pw, uv, cam, T, params.threshold_px);
    if (static_cast<int>(in.size()) < params.min_inliers) return res;
    T = refine_pose(pw, uv, in, cam, T);
    best_in = std::move(in);
  }
  res.ok = true;
  res.T_cw = T;
  res.inliers = find_inliers(pw, uv, cam, T, params.threshold_px);
  return res;
}

}  // namespace lynx
