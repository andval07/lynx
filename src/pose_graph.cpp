#include "lynx/pose_graph.hpp"

#include <Eigen/SparseCholesky>
#include <algorithm>

namespace lynx {

namespace {

using SpMat = Eigen::SparseMatrix<double>;
using VecX = Eigen::VectorXd;

// ad(xi) for xi = (rho, phi): [hat(phi) hat(rho); 0 hat(phi)].
Mat6 small_adjoint(const Vec6& xi) {
  Mat6 a = Mat6::Zero();
  a.block<3, 3>(0, 0) = hat(xi.tail<3>());
  a.block<3, 3>(0, 3) = hat(xi.head<3>());
  a.block<3, 3>(3, 3) = hat(xi.tail<3>());
  return a;
}

Vec6 edge_error(const PoseEdge& e, const std::vector<SE3>& T) {
  return (e.T_ij.inverse() * T[e.i].inverse() * T[e.j]).log();
}

double total_cost(const PoseGraph& g, const std::vector<SE3>& T) {
  double cost = 0;
  for (const auto& e : g.edges) {
    const Vec6 r = edge_error(e, T);
    cost += r.dot(e.information * r);
  }
  return cost;
}

struct Normal {
  std::vector<Eigen::Triplet<double>> H;
  VecX g;
};

// With T_w <- exp(d) T_w, the error moves as e + J_r^-1(e) Ad(T_wj^-1) (dj - di).
// J_r^-1(e) ~ I + ad(e) / 2 is enough for the small errors of a pose graph.
Normal build_normal(const PoseGraph& g, const std::vector<SE3>& T,
                    const std::vector<int>& var, int nv) {
  Normal n;
  n.g = VecX::Zero(6 * nv);
  for (const auto& e : g.edges) {
    const Vec6 r = edge_error(e, T);
    const Mat6 Jj = (Mat6::Identity() + 0.5 * small_adjoint(r)) *
                    T[e.j].inverse().adjoint();
    const int idx[2] = {var[e.i], var[e.j]};
    const Mat6 J[2] = {-Jj, Jj};
    for (int a = 0; a < 2; ++a) {
      if (idx[a] < 0) continue;
      n.g.segment<6>(6 * idx[a]) += J[a].transpose() * e.information * r;
      for (int b = 0; b < 2; ++b) {
        if (idx[b] < 0) continue;
        const Mat6 block = J[a].transpose() * e.information * J[b];
        for (int r6 = 0; r6 < 6; ++r6) {
          for (int c6 = 0; c6 < 6; ++c6) {
            n.H.emplace_back(6 * idx[a] + r6, 6 * idx[b] + c6, block(r6, c6));
          }
        }
      }
    }
  }
  return n;
}

}  // namespace

PoseGraphResult optimize_pose_graph(const PoseGraph& g,
                                    const PoseGraphOptions& opt) {
  PoseGraphResult res;
  res.T_w = g.T_w;
  std::vector<int> var(g.T_w.size(), -1);
  int nv = 0;
  for (size_t i = 0; i < g.T_w.size(); ++i) {
    if (!g.fixed[i]) var[i] = nv++;
  }
  double cost = total_cost(g, res.T_w);
  res.initial_cost = res.final_cost = cost;
  if (nv == 0 || g.edges.empty()) return res;

  double lambda = 1e-4;
  for (int it = 0; it < opt.max_iters; ++it) {
    const Normal n = build_normal(g, res.T_w, var, nv);
    SpMat H(6 * nv, 6 * nv);
    H.setFromTriplets(n.H.begin(), n.H.end());  // duplicates are summed
    const VecX diag = H.diagonal();
    bool improved = false;
    while (!improved && lambda < 1e8) {
      SpMat A = H;
      for (int k = 0; k < 6 * nv; ++k) A.coeffRef(k, k) += lambda * diag(k) + 1e-9;
      Eigen::SimplicialLDLT<SpMat> solver(A);
      if (solver.info() != Eigen::Success) {
        lambda *= 10;
        continue;
      }
      const VecX d = solver.solve(-n.g);
      std::vector<SE3> T_new = res.T_w;
      for (size_t i = 0; i < T_new.size(); ++i) {
        if (var[i] >= 0) T_new[i] = SE3::exp(d.segment<6>(6 * var[i])) * T_new[i];
      }
      const double c = total_cost(g, T_new);
      if (c < cost) {
        const double rel = (cost - c) / std::max(cost, 1e-30);
        res.T_w = std::move(T_new);
        cost = c;
        lambda = std::max(lambda / 10, 1e-9);
        improved = true;
        res.iterations = it + 1;
        if (rel < 1e-8 || d.squaredNorm() < 1e-16) it = opt.max_iters;
      } else {
        lambda *= 10;
      }
    }
    if (!improved) break;
  }
  res.final_cost = cost;
  return res;
}

}  // namespace lynx
