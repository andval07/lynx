#include "lynx/ba.hpp"

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>

namespace lynx {

namespace {

using Mat36 = Eigen::Matrix<double, 3, 6>;
using Mat63 = Eigen::Matrix<double, 6, 3>;
using MatX = Eigen::MatrixXd;
using VecX = Eigen::VectorXd;

// Residual and Jacobians of one observation. Mono observations use the first
// two rows only, the third row is zero.
struct Linearized {
  bool valid = false;
  Vec3 e = Vec3::Zero();
  Mat36 Jc = Mat36::Zero();  // d e / d xi, left perturbation of T_cw
  Mat3 Jp = Mat3::Zero();    // d e / d p_w
};

bool is_stereo(const BaObservation& o) { return o.u_right >= 0; }

double huber_threshold(const BaObservation& o, const BaOptions& opt) {
  return is_stereo(o) ? opt.huber_stereo : opt.huber_mono;
}

Vec3 residual(const BaObservation& o, const Vec3& pc, const Camera& cam,
              double bf, bool* valid) {
  *valid = pc.z() > 0;
  if (!*valid) return Vec3::Zero();
  const Vec2 uv = cam.project(pc);
  Vec3 e(uv.x() - o.uv.x(), uv.y() - o.uv.y(), 0);
  if (is_stereo(o)) e.z() = uv.x() - bf / pc.z() - o.u_right;
  return e / o.sigma;
}

Linearized linearize(const BaObservation& o, const SE3& T, const Vec3& pw,
                     const Camera& cam, double bf) {
  Linearized l;
  const Vec3 pc = T * pw;
  l.e = residual(o, pc, cam, bf, &l.valid);
  if (!l.valid) return l;
  const double iz = 1.0 / pc.z(), iz2 = iz * iz;
  Mat3 Jproj = Mat3::Zero();
  Jproj.row(0) << cam.fx * iz, 0, -cam.fx * pc.x() * iz2;
  Jproj.row(1) << 0, cam.fy * iz, -cam.fy * pc.y() * iz2;
  if (is_stereo(o)) {
    Jproj.row(2) << cam.fx * iz, 0, -cam.fx * pc.x() * iz2 + bf * iz2;
  }
  Mat36 Jt;
  Jt << Mat3::Identity(), -hat(pc);
  Jproj /= o.sigma;
  l.Jc = Jproj * Jt;
  l.Jp = Jproj * T.R();
  return l;
}

double huber_cost(double r, double k) {
  return r <= k ? r * r : 2 * k * r - k * k;
}

double huber_weight(double r, double k) { return r <= k ? 1.0 : k / r; }

double total_cost(const BaProblem& p, const std::vector<SE3>& T,
                  const std::vector<Vec3>& pts, const Camera& cam,
                  const BaOptions& opt) {
  double cost = 0;
  for (const auto& o : p.obs) {
    bool valid = false;
    const Vec3 e = residual(o, T[o.pose] * pts[o.point], cam, opt.bf, &valid);
    if (valid) cost += huber_cost(e.norm(), huber_threshold(o, opt));
  }
  return cost;
}

// Normal equations of one LM iteration, split in camera and point blocks.
struct Normal {
  MatX Hcc;                  // 6 nc x 6 nc
  VecX gc;
  std::vector<Mat3> Hpp;     // one 3x3 block per point
  std::vector<Vec3> gp;
  std::vector<Mat63> W;      // per observation, camera-point coupling
  std::vector<bool> active;  // observation contributes
};

Normal build_normal(const BaProblem& p, const std::vector<SE3>& T,
                    const std::vector<Vec3>& pts, const std::vector<int>& var,
                    int nc, const Camera& cam, const BaOptions& opt) {
  Normal n;
  n.Hcc = MatX::Zero(6 * nc, 6 * nc);
  n.gc = VecX::Zero(6 * nc);
  n.Hpp.assign(pts.size(), Mat3::Zero());
  n.gp.assign(pts.size(), Vec3::Zero());
  n.W.assign(p.obs.size(), Mat63::Zero());
  n.active.assign(p.obs.size(), false);
  for (size_t k = 0; k < p.obs.size(); ++k) {
    const auto& o = p.obs[k];
    const Linearized l = linearize(o, T[o.pose], pts[o.point], cam, opt.bf);
    if (!l.valid) continue;
    n.active[k] = true;
    const double w = huber_weight(l.e.norm(), huber_threshold(o, opt));
    n.Hpp[o.point] += w * l.Jp.transpose() * l.Jp;
    n.gp[o.point] += w * l.Jp.transpose() * l.e;
    const int c = var[o.pose];
    if (c < 0) continue;
    n.Hcc.block<6, 6>(6 * c, 6 * c) += w * l.Jc.transpose() * l.Jc;
    n.gc.segment<6>(6 * c) += w * l.Jc.transpose() * l.e;
    n.W[k] = w * l.Jc.transpose() * l.Jp;
  }
  return n;
}

}  // namespace

BaResult bundle_adjust(const BaProblem& p, const Camera& cam,
                       const BaOptions& opt) {
  BaResult res;
  res.T_cw = p.T_cw;
  res.points = p.points;

  // Free poses get a variable index, fixed poses -1.
  std::vector<int> var(p.T_cw.size(), -1);
  int nc = 0;
  for (size_t i = 0; i < p.T_cw.size(); ++i) {
    if (!p.fixed[i]) var[i] = nc++;
  }
  std::vector<std::vector<int>> obs_of_point(p.points.size());
  for (size_t k = 0; k < p.obs.size(); ++k) {
    obs_of_point[p.obs[k].point].push_back(static_cast<int>(k));
  }

  double cost = total_cost(p, res.T_cw, res.points, cam, opt);
  res.initial_cost = cost;
  double lambda = 1e-4;
  for (int it = 0; it < opt.max_iters; ++it) {
    const Normal n = build_normal(p, res.T_cw, res.points, var, nc, cam, opt);
    bool improved = false;
    while (!improved && lambda < 1e8) {
      // Schur complement: S dc = b, then each point from its own 3x3 block.
      MatX S = n.Hcc;
      S.diagonal() += lambda * n.Hcc.diagonal() + VecX::Constant(6 * nc, 1e-9);
      VecX b = -n.gc;
      std::vector<Mat3> Vinv(res.points.size(), Mat3::Zero());
      for (size_t j = 0; j < res.points.size(); ++j) {
        if (obs_of_point[j].empty()) continue;
        Mat3 V = n.Hpp[j];
        V.diagonal() += lambda * n.Hpp[j].diagonal() + Vec3::Constant(1e-9);
        Vinv[j] = V.inverse();
        for (int a : obs_of_point[j]) {
          const int ca = var[p.obs[a].pose];
          if (!n.active[a] || ca < 0) continue;
          const Mat63 WV = n.W[a] * Vinv[j];
          b.segment<6>(6 * ca) += WV * n.gp[j];
          for (int c : obs_of_point[j]) {
            const int cb = var[p.obs[c].pose];
            if (!n.active[c] || cb < 0) continue;
            S.block<6, 6>(6 * ca, 6 * cb) -= WV * n.W[c].transpose();
          }
        }
      }
      const VecX dc = nc > 0 ? VecX(S.ldlt().solve(b)) : VecX();

      std::vector<SE3> T_new = res.T_cw;
      for (size_t i = 0; i < T_new.size(); ++i) {
        if (var[i] >= 0) T_new[i] = SE3::exp(dc.segment<6>(6 * var[i])) * T_new[i];
      }
      std::vector<Vec3> pts_new = res.points;
      double step2 = nc > 0 ? dc.squaredNorm() : 0;
      for (size_t j = 0; j < pts_new.size(); ++j) {
        if (obs_of_point[j].empty()) continue;
        Vec3 r = -n.gp[j];
        for (int a : obs_of_point[j]) {
          const int ca = var[p.obs[a].pose];
          if (n.active[a] && ca >= 0) {
            r -= n.W[a].transpose() * dc.segment<6>(6 * ca);
          }
        }
        const Vec3 dp = Vinv[j] * r;
        pts_new[j] += dp;
        step2 += dp.squaredNorm();
      }

      const double c = total_cost(p, T_new, pts_new, cam, opt);
      if (c < cost) {
        const double rel = (cost - c) / cost;
        res.T_cw = std::move(T_new);
        res.points = std::move(pts_new);
        cost = c;
        lambda = std::max(lambda / 10, 1e-9);
        improved = true;
        res.iterations = it + 1;
        if (rel < 1e-6 || step2 < 1e-16) it = opt.max_iters;
      } else {
        lambda *= 10;
      }
    }
    if (!improved) break;
  }
  res.final_cost = cost;

  res.outlier.assign(p.obs.size(), false);
  for (size_t k = 0; k < p.obs.size(); ++k) {
    const auto& o = p.obs[k];
    bool valid = false;
    const Vec3 e = residual(o, res.T_cw[o.pose] * res.points[o.point], cam,
                            opt.bf, &valid);
    res.outlier[k] = !valid || e.norm() > huber_threshold(o, opt);
  }
  return res;
}

}  // namespace lynx
