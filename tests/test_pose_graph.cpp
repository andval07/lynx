#include <gtest/gtest.h>

#include <random>

#include "lynx/pose_graph.hpp"

using namespace lynx;

namespace {

// Poses on a circle of radius 2 m, each turned to face the direction of travel.
std::vector<SE3> circle(int n) {
  std::vector<SE3> T;
  for (int k = 0; k < n; ++k) {
    const double a = 2 * M_PI * k / n;
    Vec6 xi = Vec6::Zero();
    xi(5) = a;
    const SE3 rot = SE3::exp(xi);
    T.emplace_back(rot.R(), Vec3(2 * std::cos(a), 2 * std::sin(a), 0.1 * k));
  }
  return T;
}

Vec6 random_twist(std::mt19937& rng, double sigma_t, double sigma_r) {
  std::normal_distribution<double> nt(0, sigma_t), nr(0, sigma_r);
  Vec6 xi;
  xi << nt(rng), nt(rng), nt(rng), nr(rng), nr(rng), nr(rng);
  return xi;
}

double max_position_error(const std::vector<SE3>& a, const std::vector<SE3>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (a[i].t() - b[i].t()).norm());
  return m;
}

// Odometry chain plus one loop edge from the last pose back to the first.
PoseGraph chain_with_loop(const std::vector<SE3>& truth) {
  PoseGraph g;
  g.T_w = truth;
  g.fixed.assign(truth.size(), false);
  g.fixed[0] = true;
  const int n = static_cast<int>(truth.size());
  for (int k = 1; k < n; ++k) {
    g.edges.push_back({k - 1, k, truth[k - 1].inverse() * truth[k]});
  }
  g.edges.push_back({0, n - 1, truth[0].inverse() * truth[n - 1]});
  return g;
}

}  // namespace

TEST(PoseGraph, ExactGraphStaysPut) {
  const auto truth = circle(20);
  const PoseGraphResult r = optimize_pose_graph(chain_with_loop(truth));
  EXPECT_LT(r.initial_cost, 1e-20);
  EXPECT_LT(max_position_error(r.T_w, truth), 1e-9);
}

TEST(PoseGraph, RecoversTruthFromPerturbedStart) {
  std::mt19937 rng(3);
  const auto truth = circle(30);
  PoseGraph g = chain_with_loop(truth);
  for (size_t k = 1; k < g.T_w.size(); ++k) {
    g.T_w[k] = SE3::exp(random_twist(rng, 0.2, 0.1)) * g.T_w[k];
  }
  const PoseGraphResult r = optimize_pose_graph(g);
  EXPECT_LT(r.final_cost, 1e-12);
  EXPECT_LT(max_position_error(r.T_w, truth), 1e-5);
  EXPECT_LT(r.iterations, 15);
}

TEST(PoseGraph, LoopEdgeSpreadsDrift) {
  // Odometry with a small bias: the chain alone ends far from the start. The
  // loop edge should pull the end back and share the error along the chain.
  std::mt19937 rng(5);
  const auto truth = circle(40);
  PoseGraph g = chain_with_loop(truth);
  const int n = static_cast<int>(truth.size());
  Vec6 bias = Vec6::Zero();
  bias << 0.01, 0, 0, 0, 0, 0.01;
  for (int k = 0; k + 1 < n; ++k) {
    g.edges[k].T_ij = g.edges[k].T_ij * SE3::exp(bias + random_twist(rng, 0.002, 0.001));
  }
  // Start from dead reckoning, as the odometry would give it.
  for (int k = 1; k < n; ++k) g.T_w[k] = g.T_w[k - 1] * g.edges[k - 1].T_ij;
  const double before = (g.T_w[n - 1].t() - truth[n - 1].t()).norm();

  const PoseGraphResult r = optimize_pose_graph(g);
  const double after = (r.T_w[n - 1].t() - truth[n - 1].t()).norm();
  EXPECT_LT(r.final_cost, r.initial_cost);
  EXPECT_LT(after, 0.1 * before);
}

TEST(PoseGraph, FixedNodesDoNotMove) {
  std::mt19937 rng(7);
  const auto truth = circle(10);
  PoseGraph g = chain_with_loop(truth);
  g.fixed[4] = true;
  g.T_w[4] = SE3::exp(random_twist(rng, 0.1, 0.05)) * g.T_w[4];
  const SE3 held = g.T_w[4];
  const PoseGraphResult r = optimize_pose_graph(g);
  EXPECT_LT((r.T_w[4].t() - held.t()).norm(), 1e-12);
  EXPECT_LT((r.T_w[0].t() - truth[0].t()).norm(), 1e-12);
}
