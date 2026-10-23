#include "lynx/trajectory.hpp"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <tuple>

namespace lynx {

std::vector<std::pair<int, int>> associate(const Trajectory& a,
                                           const Trajectory& b, double max_dt) {
  // Candidate pairs within max_dt, then greedy by time difference.
  std::vector<std::tuple<double, int, int>> cand;
  size_t lo = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    while (lo < b.size() && b[lo].stamp < a[i].stamp - max_dt) ++lo;
    for (size_t j = lo; j < b.size() && b[j].stamp <= a[i].stamp + max_dt; ++j) {
      cand.emplace_back(std::abs(a[i].stamp - b[j].stamp), static_cast<int>(i),
                        static_cast<int>(j));
    }
  }
  std::sort(cand.begin(), cand.end());
  std::vector<bool> used_a(a.size(), false), used_b(b.size(), false);
  std::vector<std::pair<int, int>> out;
  for (const auto& [dt, i, j] : cand) {
    if (used_a[i] || used_b[j]) continue;
    used_a[i] = used_b[j] = true;
    out.emplace_back(i, j);
  }
  std::sort(out.begin(), out.end());
  return out;
}

SE3 align_umeyama(const std::vector<Vec3>& src, const std::vector<Vec3>& dst) {
  const auto n = static_cast<double>(src.size());
  Vec3 ms = Vec3::Zero(), md = Vec3::Zero();
  for (size_t i = 0; i < src.size(); ++i) {
    ms += src[i];
    md += dst[i];
  }
  ms /= n;
  md /= n;
  Mat3 H = Mat3::Zero();
  for (size_t i = 0; i < src.size(); ++i) {
    H += (dst[i] - md) * (src[i] - ms).transpose();
  }
  Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 D = Mat3::Identity();
  if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0) {
    D(2, 2) = -1;
  }
  const Mat3 R = svd.matrixU() * D * svd.matrixV().transpose();
  return SE3(R, md - R * ms);
}

double ate_rmse(const Trajectory& estimate, const Trajectory& ground_truth) {
  const auto pairs = associate(estimate, ground_truth);
  if (pairs.size() < 3) return -1;
  std::vector<Vec3> src, dst;
  for (const auto& [i, j] : pairs) {
    src.push_back(estimate[i].T_wc.t());
    dst.push_back(ground_truth[j].T_wc.t());
  }
  const SE3 T = align_umeyama(src, dst);
  double sum = 0;
  for (size_t k = 0; k < src.size(); ++k) {
    sum += (T * src[k] - dst[k]).squaredNorm();
  }
  return std::sqrt(sum / static_cast<double>(src.size()));
}

}  // namespace lynx
