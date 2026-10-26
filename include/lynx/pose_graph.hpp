#pragma once

#include <vector>

#include "lynx/se3.hpp"

namespace lynx {

// Relative pose measurement between nodes i and j: T_ij ~ T_wi^-1 T_wj.
struct PoseEdge {
  int i = 0;
  int j = 0;
  SE3 T_ij;
  Mat6 information = Mat6::Identity();
};

struct PoseGraph {
  std::vector<SE3> T_w;     // world-from-node poses
  std::vector<bool> fixed;  // same size as T_w
  std::vector<PoseEdge> edges;
};

struct PoseGraphOptions {
  int max_iters = 20;
};

struct PoseGraphResult {
  std::vector<SE3> T_w;
  double initial_cost = 0;
  double final_cost = 0;
  int iterations = 0;
};

// Levenberg-Marquardt on the edge errors e = log(T_ij^-1 T_wi^-1 T_wj), with a
// left perturbation on every free T_w and a sparse Cholesky solve.
PoseGraphResult optimize_pose_graph(const PoseGraph& graph,
                                    const PoseGraphOptions& opt = {});

}  // namespace lynx
