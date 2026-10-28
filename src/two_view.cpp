#include "lynx/two_view.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "lynx/ba.hpp"
#include "lynx/geometry.hpp"

namespace lynx {

namespace {

constexpr double kChi2OneDof = 3.841;  // 95%, epipolar distance
constexpr double kChi2TwoDof = 5.991;  // 95%, point transfer
constexpr int kSample = 8;

using Mat9 = Eigen::Matrix<double, Eigen::Dynamic, 9>;

// Right null vector of A as a row-major 3x3 matrix.
Mat3 null_matrix(const Mat9& A) {
  Eigen::JacobiSVD<Mat9> svd(A, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 1> h = svd.matrixV().col(8);
  Mat3 M;
  M << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
  return M;
}

// Hartley normalization: centroid at 0, mean distance sqrt(2).
Mat3 normalize_points(const std::vector<Vec2>& x, std::vector<Vec2>& out) {
  Vec2 c = Vec2::Zero();
  for (const Vec2& p : x) c += p;
  c /= static_cast<double>(x.size());
  double d = 0;
  for (const Vec2& p : x) d += (p - c).norm();
  const double k = std::sqrt(2.0) * x.size() / std::max(d, 1e-12);
  out.clear();
  for (const Vec2& p : x) out.push_back(k * (p - c));
  Mat3 T;
  T << k, 0, -k * c.x(), 0, k, -k * c.y(), 0, 0, 1;
  return T;
}

struct Model {
  Mat3 M = Mat3::Zero();
  double score = 0;
  std::vector<bool> inlier;
};

// Errors are measured in pixels: normalized differences times the focal.
double f_mean(const Camera& cam) { return 0.5 * (cam.fx + cam.fy); }

// Squared epipolar distance in both images over sigma^2, as in ORB-SLAM.
Model score_essential(const Mat3& E, const std::vector<Vec2>& x1,
                      const std::vector<Vec2>& x2, double f, double sigma) {
  Model m{E, 0, std::vector<bool>(x1.size(), false)};
  const double inv_s2 = 1.0 / (sigma * sigma);
  for (size_t i = 0; i < x1.size(); ++i) {
    const Vec3 a = x1[i].homogeneous(), b = x2[i].homogeneous();
    const Vec3 l2 = E * a, l1 = E.transpose() * b;
    const double num = b.dot(l2);
    const double c2 = f * f * num * num / l2.head<2>().squaredNorm() * inv_s2;
    const double c1 = f * f * num * num / l1.head<2>().squaredNorm() * inv_s2;
    if (c1 > kChi2OneDof || c2 > kChi2OneDof) continue;
    m.score += 2 * kChi2TwoDof - c1 - c2;
    m.inlier[i] = true;
  }
  return m;
}

// Symmetric transfer error over sigma^2.
Model score_homography(const Mat3& H, const std::vector<Vec2>& x1,
                       const std::vector<Vec2>& x2, double f, double sigma) {
  Model m{H, 0, std::vector<bool>(x1.size(), false)};
  const Mat3 Hinv = H.inverse();
  const double inv_s2 = 1.0 / (sigma * sigma);
  for (size_t i = 0; i < x1.size(); ++i) {
    const Vec2 p2 = (H * x1[i].homogeneous()).hnormalized();
    const Vec2 p1 = (Hinv * x2[i].homogeneous()).hnormalized();
    const double c2 = f * f * (p2 - x2[i]).squaredNorm() * inv_s2;
    const double c1 = f * f * (p1 - x1[i]).squaredNorm() * inv_s2;
    if (c1 > kChi2TwoDof || c2 > kChi2TwoDof) continue;
    m.score += 2 * kChi2TwoDof - c1 - c2;
    m.inlier[i] = true;
  }
  return m;
}

template <typename T>
std::vector<T> pick(const std::vector<T>& v, const std::vector<bool>& keep) {
  std::vector<T> out;
  for (size_t i = 0; i < v.size(); ++i) {
    if (keep[i]) out.push_back(v[i]);
  }
  return out;
}

// Same 8-point samples for both models, then one refit on the inliers.
void ransac_models(const std::vector<Vec2>& x1, const std::vector<Vec2>& x2,
                   double f, const TwoViewParams& p, std::mt19937& rng,
                   Model& best_e, Model& best_h) {
  std::vector<int> idx(x1.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<Vec2> s1(kSample), s2(kSample);
  for (int it = 0; it < p.max_iters; ++it) {
    for (int k = 0; k < kSample; ++k) {
      std::uniform_int_distribution<int> d(k, static_cast<int>(idx.size()) - 1);
      std::swap(idx[k], idx[d(rng)]);
      s1[k] = x1[idx[k]];
      s2[k] = x2[idx[k]];
    }
    Model e = score_essential(essential_8point(s1, s2), x1, x2, f, p.sigma_px);
    if (e.score > best_e.score) best_e = std::move(e);
    Model h = score_homography(homography_dlt(s1, s2), x1, x2, f, p.sigma_px);
    if (h.score > best_h.score) best_h = std::move(h);
  }
  const auto refit = [&](Model& m, bool essential) {
    const auto in1 = pick(x1, m.inlier), in2 = pick(x2, m.inlier);
    if (static_cast<int>(in1.size()) < kSample) return;
    Model r = essential
                  ? score_essential(essential_8point(in1, in2), x1, x2, f, p.sigma_px)
                  : score_homography(homography_dlt(in1, in2), x1, x2, f, p.sigma_px);
    if (r.score > m.score) m = std::move(r);
  };
  refit(best_e, true);
  refit(best_h, false);
}

struct Check {
  int good = 0;
  double parallax_deg = 0;
  std::vector<Vec3> points;
  std::vector<bool> ok;
};

// Triangulates the inliers with T_21 and keeps the points in front of both
// cameras with a small reprojection error.
Check check_pose(const SE3& T21, const std::vector<Vec2>& x1,
                 const std::vector<Vec2>& x2, const std::vector<bool>& inlier,
                 double f, double sigma) {
  Check c{0, 0, std::vector<Vec3>(x1.size(), Vec3::Zero()),
          std::vector<bool>(x1.size(), false)};
  const Camera unit{1, 1, 0, 0};
  const Mat34 P1 = projection_matrix(unit, SE3());
  const Mat34 P2 = projection_matrix(unit, T21);
  const Vec3 c2 = T21.inverse().t();
  const double max_err2 = 4.0 * sigma * sigma / (f * f);
  std::vector<double> parallax;
  for (size_t i = 0; i < x1.size(); ++i) {
    if (!inlier[i]) continue;
    const Vec3 X = triangulate(P1, P2, x1[i], x2[i]);
    if (!X.allFinite()) continue;
    const Vec3 X2 = T21 * X;
    if (X.z() <= 0 || X2.z() <= 0) continue;
    if ((X.head<2>() / X.z() - x1[i]).squaredNorm() > max_err2) continue;
    if ((X2.head<2>() / X2.z() - x2[i]).squaredNorm() > max_err2) continue;
    const double cosp = X.normalized().dot((X - c2).normalized());
    parallax.push_back(std::acos(std::clamp(cosp, -1.0, 1.0)) * 180.0 / M_PI);
    c.points[i] = X;
    c.ok[i] = true;
    ++c.good;
  }
  if (!parallax.empty()) {
    auto mid = parallax.begin() + parallax.size() / 2;
    std::nth_element(parallax.begin(), mid, parallax.end());
    c.parallax_deg = *mid;
  }
  return c;
}

// Two-view BA in pixels with camera 1 fixed, then back to |t| = 1. The linear
// E is noisy when the parallax is small, this fixes most of it.
SE3 refine_two_view(const SE3& T21, const Check& c, const std::vector<Vec2>& uv1,
                    const std::vector<Vec2>& uv2, const Camera& cam) {
  BaProblem pb;
  pb.T_cw = {SE3(), T21};
  pb.fixed = {true, false};
  for (size_t i = 0; i < c.ok.size(); ++i) {
    if (!c.ok[i]) continue;
    const int j = static_cast<int>(pb.points.size());
    pb.points.push_back(c.points[i]);
    pb.obs.push_back({0, j, uv1[i]});
    pb.obs.push_back({1, j, uv2[i]});
  }
  BaOptions opt;
  opt.max_iters = 20;
  const SE3 T = bundle_adjust(pb, cam, opt).T_cw[1];
  return SE3(T.R(), T.t().normalized());
}

}  // namespace

Mat3 essential_8point(const std::vector<Vec2>& x1_in, const std::vector<Vec2>& x2_in) {
  std::vector<Vec2> x1, x2;
  const Mat3 T1 = normalize_points(x1_in, x1), T2 = normalize_points(x2_in, x2);
  Mat9 A(x1.size(), 9);
  for (size_t i = 0; i < x1.size(); ++i) {
    const double u1 = x1[i].x(), v1 = x1[i].y(), u2 = x2[i].x(), v2 = x2[i].y();
    A.row(i) << u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1;
  }
  const Mat3 F = T2.transpose() * null_matrix(A) * T1;
  Eigen::JacobiSVD<Mat3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  return svd.matrixU() * Vec3(1, 1, 0).asDiagonal() * svd.matrixV().transpose();
}

Mat3 homography_dlt(const std::vector<Vec2>& x1_in, const std::vector<Vec2>& x2_in) {
  std::vector<Vec2> x1, x2;
  const Mat3 T1 = normalize_points(x1_in, x1), T2 = normalize_points(x2_in, x2);
  Mat9 A(2 * x1.size(), 9);
  for (size_t i = 0; i < x1.size(); ++i) {
    const double x = x1[i].x(), y = x1[i].y(), u = x2[i].x(), v = x2[i].y();
    A.row(2 * i) << -x, -y, -1, 0, 0, 0, u * x, u * y, u;
    A.row(2 * i + 1) << 0, 0, 0, -x, -y, -1, v * x, v * y, v;
  }
  return T2.inverse() * null_matrix(A) * T1;
}

std::vector<SE3> decompose_essential(const Mat3& E) {
  Eigen::JacobiSVD<Mat3> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 U = svd.matrixU(), V = svd.matrixV();
  if (U.determinant() < 0) U = -U;
  if (V.determinant() < 0) V = -V;
  Mat3 W;
  W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  const Mat3 Ra = U * W * V.transpose(), Rb = U * W.transpose() * V.transpose();
  const Vec3 t = U.col(2);
  return {SE3(Ra, t), SE3(Ra, -t), SE3(Rb, t), SE3(Rb, -t)};
}

std::vector<SE3> decompose_homography(const Mat3& H_in) {
  // H is known up to scale and sign: normalize so that sigma_2 = 1 and try
  // both signs. The chirality check later keeps the right one.
  Eigen::JacobiSVD<Mat3> svd_h(H_in);
  const Mat3 Hn = H_in / svd_h.singularValues()(1);
  std::vector<SE3> out;
  for (const double sign : {1.0, -1.0}) {
    const Mat3 H = sign * Hn;
    Eigen::SelfAdjointEigenSolver<Mat3> eig(H.transpose() * H);
    const Vec3 s2 = eig.eigenvalues();  // ascending: sigma_3^2, 1, sigma_1^2
    const Mat3 V = eig.eigenvectors();
    const double gap = s2(2) - s2(0);
    if (gap < 1e-9) return {};  // pure rotation, no translation to recover
    const Vec3 v1 = V.col(2), v2 = V.col(1), v3 = V.col(0);
    const double a = std::sqrt(std::max(0.0, 1.0 - s2(0)));
    const double b = std::sqrt(std::max(0.0, s2(2) - 1.0));
    for (const double k : {1.0, -1.0}) {
      const Vec3 u = (a * v1 + k * b * v3) / std::sqrt(gap);
      Mat3 Um, Wm;
      Um << v2, u, hat(v2) * u;
      Wm << H * v2, H * u, hat(H * v2) * (H * u);
      const Mat3 R = Wm * Um.transpose();
      const Vec3 n = hat(v2) * u;
      const Vec3 t = (H - R) * n;
      out.emplace_back(R, t);
      out.emplace_back(R, -t);  // the same with -n
    }
  }
  return out;
}

TwoViewResult initialize_two_view(const std::vector<Vec2>& uv1,
                                  const std::vector<Vec2>& uv2,
                                  const Camera& cam, const TwoViewParams& p,
                                  std::mt19937& rng) {
  TwoViewResult res;
  res.points.assign(uv1.size(), Vec3::Zero());
  res.good.assign(uv1.size(), false);
  if (uv1.size() < static_cast<size_t>(std::max(kSample, p.min_points))) return res;
  std::vector<Vec2> x1, x2;
  for (size_t i = 0; i < uv1.size(); ++i) {
    x1.push_back(cam.normalize(uv1[i]));
    x2.push_back(cam.normalize(uv2[i]));
  }
  const double f = f_mean(cam);
  Model e, h;
  ransac_models(x1, x2, f, p, rng, e, h);
  if (e.score + h.score <= 0) return res;
  res.homography = h.score / (h.score + e.score) > p.homography_ratio;
  const Model& m = res.homography ? h : e;
  const auto cands = res.homography ? decompose_homography(m.M) : decompose_essential(m.M);

  const int n_inliers = static_cast<int>(std::count(m.inlier.begin(), m.inlier.end(), true));
  Check best, second;
  SE3 best_T;
  for (const SE3& T : cands) {
    if (T.t().norm() < 1e-12) continue;
    const SE3 Tu(T.R(), T.t().normalized());
    Check c = check_pose(Tu, x1, x2, m.inlier, f, p.sigma_px);
    if (c.good > best.good) {
      second = std::move(best);
      best = std::move(c);
      best_T = Tu;
    } else if (c.good > second.good) {
      second = std::move(c);
    }
  }
  // A clear winner, most inliers explained, and enough parallax.
  const int min_good = std::max(p.min_points, static_cast<int>(0.9 * n_inliers));
  if (best.good < min_good || second.good > 0.75 * best.good) return res;
  if (best.parallax_deg < p.min_parallax_deg) return res;
  // Refine, then take the inliers again from the refined E: the first
  // linear E is noisy and misses many good matches.
  std::vector<bool> inlier = m.inlier;
  for (int k = 0; k < 3; ++k) {
    best_T = refine_two_view(best_T, best, uv1, uv2, cam);
    const Mat3 E = hat(best_T.t()) * best_T.R();
    inlier = score_essential(E, x1, x2, f, p.sigma_px).inlier;
    best = check_pose(best_T, x1, x2, inlier, f, p.sigma_px);
  }
  if (best.good < p.min_points) return res;
  res.ok = true;
  res.T_21 = best_T;
  res.points = std::move(best.points);
  res.good = std::move(best.ok);
  res.parallax_deg = best.parallax_deg;
  return res;
}

}  // namespace lynx
