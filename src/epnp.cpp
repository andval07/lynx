// EPnP: every world point is a fixed weighted sum of 4 control points, so the
// pose problem becomes finding the 4 control points in the camera frame
// (12 unknowns). They lie in the null space of a 2n x 12 linear system, and
// the betas that mix the null vectors come from the distances between control
// points, which do not change with the pose.

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <array>
#include <cmath>

#include "lynx/geometry.hpp"

namespace lynx {

namespace {

using Mat12 = Eigen::Matrix<double, 12, 12>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Vec4 = Eigen::Vector4d;
using L6x10 = Eigen::Matrix<double, 6, 10>;
using Vec6d = Eigen::Matrix<double, 6, 1>;

constexpr std::array<std::array<int, 2>, 6> kPairs = {
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

struct Controls {
  std::array<Vec3, 4> cw;
  std::vector<Vec4> alphas;
};

// Control points: centroid plus the principal axes, scaled by their spread.
Controls choose_controls(const std::vector<Vec3>& pw) {
  const auto n = static_cast<double>(pw.size());
  Controls c;
  Vec3 mean = Vec3::Zero();
  for (const auto& p : pw) mean += p;
  mean /= n;

  Mat3 cov = Mat3::Zero();
  for (const auto& p : pw) cov += (p - mean) * (p - mean).transpose();
  cov /= n;
  Eigen::SelfAdjointEigenSolver<Mat3> es(cov);
  const double lmax = es.eigenvalues().maxCoeff();

  c.cw[0] = mean;
  Mat3 B;  // columns are cw[k] - cw[0]
  for (int k = 0; k < 3; ++k) {
    // Floor the spread so that a planar scene still gives an invertible basis.
    const double s = std::sqrt(std::max(es.eigenvalues()(k), 1e-6 * lmax));
    c.cw[k + 1] = mean + s * es.eigenvectors().col(k);
    B.col(k) = c.cw[k + 1] - mean;
  }
  const Mat3 Binv = B.inverse();
  c.alphas.reserve(pw.size());
  for (const auto& p : pw) {
    const Vec3 a = Binv * (p - mean);
    c.alphas.emplace_back(1.0 - a.sum(), a(0), a(1), a(2));
  }
  return c;
}

// Rows of L multiply the products
// [b11, b12, b22, b13, b23, b33, b14, b24, b34, b44].
L6x10 compute_L(const std::array<Vec12, 4>& v) {
  L6x10 L;
  for (int r = 0; r < 6; ++r) {
    const auto [a, b] = kPairs[r];
    std::array<Vec3, 4> d;
    for (int k = 0; k < 4; ++k) {
      d[k] = v[k].segment<3>(3 * a) - v[k].segment<3>(3 * b);
    }
    L.row(r) << d[0].dot(d[0]), 2 * d[0].dot(d[1]), d[1].dot(d[1]),
        2 * d[0].dot(d[2]), 2 * d[1].dot(d[2]), d[2].dot(d[2]),
        2 * d[0].dot(d[3]), 2 * d[1].dot(d[3]), 2 * d[2].dot(d[3]),
        d[3].dot(d[3]);
  }
  return L;
}

Eigen::Matrix<double, 10, 1> products(const Vec4& b) {
  Eigen::Matrix<double, 10, 1> p;
  p << b(0) * b(0), b(0) * b(1), b(1) * b(1), b(0) * b(2), b(1) * b(2),
      b(2) * b(2), b(0) * b(3), b(1) * b(3), b(2) * b(3), b(3) * b(3);
  return p;
}

// Initial betas, as in the paper: solve L for a subset of the products, then
// read the betas back from the squares. N is the number of null vectors used.
Vec4 betas_approx(const L6x10& L, const Vec6d& rho, int N) {
  Vec4 b = Vec4::Zero();
  if (N == 1) {
    Eigen::Matrix<double, 6, 4> A;
    A << L.col(0), L.col(1), L.col(3), L.col(6);
    const Vec4 x = A.colPivHouseholderQr().solve(rho);
    const double s = x(0) < 0 ? -1.0 : 1.0;
    b(0) = std::sqrt(std::abs(x(0)));
    for (int k = 1; k < 4; ++k) b(k) = s * x(k) / b(0);
  } else if (N == 2) {
    Eigen::Matrix<double, 6, 3> A;
    A << L.col(0), L.col(1), L.col(2);
    const Vec3 x = A.colPivHouseholderQr().solve(rho);
    const double s = x(0) < 0 ? -1.0 : 1.0;
    b(0) = std::sqrt(std::abs(x(0)));
    b(1) = s * x(2) > 0 ? std::sqrt(std::abs(x(2))) : 0.0;
    if (s * x(1) < 0) b(0) = -b(0);
  } else {
    Eigen::Matrix<double, 6, 5> A;
    A << L.col(0), L.col(1), L.col(2), L.col(3), L.col(4);
    const Eigen::Matrix<double, 5, 1> x = A.colPivHouseholderQr().solve(rho);
    const double s = x(0) < 0 ? -1.0 : 1.0;
    b(0) = std::sqrt(std::abs(x(0)));
    b(1) = s * x(2) > 0 ? std::sqrt(std::abs(x(2))) : 0.0;
    if (s * x(1) < 0) b(0) = -b(0);
    b(2) = s * x(3) / b(0);
  }
  return b;
}

// Gauss-Newton on || L * products(b) - rho ||^2.
Vec4 refine_betas(const L6x10& L, const Vec6d& rho, Vec4 b) {
  for (int it = 0; it < 5; ++it) {
    Eigen::Matrix<double, 6, 4> J;
    for (int r = 0; r < 6; ++r) {
      const auto l = L.row(r);
      J(r, 0) = 2 * l(0) * b(0) + l(1) * b(1) + l(3) * b(2) + l(6) * b(3);
      J(r, 1) = l(1) * b(0) + 2 * l(2) * b(1) + l(4) * b(2) + l(7) * b(3);
      J(r, 2) = l(3) * b(0) + l(4) * b(1) + 2 * l(5) * b(2) + l(8) * b(3);
      J(r, 3) = l(6) * b(0) + l(7) * b(1) + l(8) * b(2) + 2 * l(9) * b(3);
    }
    const Vec6d r = rho - L * products(b);
    b += J.colPivHouseholderQr().solve(r);
  }
  return b;
}

// Absolute orientation: R, t that minimise sum |pc - (R pw + t)|^2.
SE3 procrustes(const std::vector<Vec3>& pw, const std::vector<Vec3>& pc) {
  const auto n = static_cast<double>(pw.size());
  Vec3 mw = Vec3::Zero(), mc = Vec3::Zero();
  for (size_t i = 0; i < pw.size(); ++i) {
    mw += pw[i];
    mc += pc[i];
  }
  mw /= n;
  mc /= n;
  Mat3 H = Mat3::Zero();
  for (size_t i = 0; i < pw.size(); ++i) {
    H += (pc[i] - mc) * (pw[i] - mw).transpose();
  }
  Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 D = Mat3::Identity();
  if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0) {
    D(2, 2) = -1;
  }
  const Mat3 R = svd.matrixU() * D * svd.matrixV().transpose();
  return SE3(R, mc - R * mw);
}

double mean_reprojection(const std::vector<Vec3>& pw,
                         const std::vector<Vec2>& xn, const SE3& T) {
  double e = 0;
  for (size_t i = 0; i < pw.size(); ++i) {
    const Vec3 p = T * pw[i];
    e += (p.head<2>() / p.z() - xn[i]).norm();
  }
  return e / static_cast<double>(pw.size());
}

}  // namespace

std::optional<SE3> epnp(const std::vector<Vec3>& pw, const std::vector<Vec2>& uv,
                        const Camera& cam) {
  const size_t n = pw.size();
  if (n < 6 || uv.size() != n) return std::nullopt;

  const Controls c = choose_controls(pw);
  std::vector<Vec2> xn(n);
  for (size_t i = 0; i < n; ++i) xn[i] = cam.normalize(uv[i]);

  // M^T M accumulated directly; each point adds two rows of M.
  Mat12 MtM = Mat12::Zero();
  for (size_t i = 0; i < n; ++i) {
    Vec12 r1 = Vec12::Zero(), r2 = Vec12::Zero();
    for (int j = 0; j < 4; ++j) {
      const double a = c.alphas[i](j);
      r1(3 * j) = a;
      r1(3 * j + 2) = -a * xn[i].x();
      r2(3 * j + 1) = a;
      r2(3 * j + 2) = -a * xn[i].y();
    }
    MtM += r1 * r1.transpose() + r2 * r2.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Mat12> es(MtM);
  if (es.info() != Eigen::Success) return std::nullopt;
  std::array<Vec12, 4> v;
  for (int k = 0; k < 4; ++k) v[k] = es.eigenvectors().col(k);  // smallest

  const L6x10 L = compute_L(v);
  Vec6d rho;
  for (int r = 0; r < 6; ++r) {
    rho(r) = (c.cw[kPairs[r][0]] - c.cw[kPairs[r][1]]).squaredNorm();
  }

  std::optional<SE3> best;
  double best_err = std::numeric_limits<double>::infinity();
  std::vector<Vec3> pc(n);
  for (int N = 1; N <= 3; ++N) {
    const Vec4 b = refine_betas(L, rho, betas_approx(L, rho, N));
    Vec12 x = Vec12::Zero();
    for (int k = 0; k < 4; ++k) x += b(k) * v[k];
    for (size_t i = 0; i < n; ++i) {
      pc[i] = Vec3::Zero();
      for (int j = 0; j < 4; ++j) pc[i] += c.alphas[i](j) * x.segment<3>(3 * j);
    }
    // The null space has no sign: flip so the points are in front.
    if (pc[0].z() < 0) {
      for (auto& p : pc) p = -p;
    }
    if (!pc[0].allFinite()) continue;
    const SE3 T = procrustes(pw, pc);
    const double err = mean_reprojection(pw, xn, T);
    if (err < best_err) {
      best_err = err;
      best = T;
    }
  }
  return best;
}

}  // namespace lynx
