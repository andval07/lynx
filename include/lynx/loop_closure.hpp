#pragma once

#include <random>
#include <unordered_map>
#include <vector>

#include "lynx/geometry.hpp"
#include "lynx/map.hpp"
#include "lynx/pose_graph.hpp"
#include "lynx/vocabulary.hpp"

namespace lynx {

struct LoopParams {
  int min_gap = 30;         // keyframes between the current one and a candidate
  double min_score = 0.05;  // absolute floor on the BoW score
  // A candidate must score at least this fraction of the score between the
  // current keyframe and the previous one, which sees almost the same scene.
  double min_score_ratio = 0.75;
  int max_candidates = 3;   // best scoring ones that get a geometric check
  float ratio = 0.8f;       // Lowe ratio for the loop matches
  int min_inliers = 50;
  RansacParams ransac;
  int covis_min_shared = 100;  // points shared to add a covisibility edge
  PoseGraphOptions pgo;
};

struct LoopEdge {
  int cand = 0;
  int cur = 0;
  SE3 T_cand_cur;
  int inliers = 0;
  double score = 0;
};

// Loop detection with a bag of words database of the keyframes, geometric
// check with PnP against the map points of the candidate, and correction of
// the whole map with a pose graph.
class LoopCloser {
 public:
  LoopCloser(Vocabulary voc, const Camera& cam, LoopParams p = {});

  // Adds keyframe `id` (the newest one in `map`) to the database and looks
  // for a loop. When one is verified it optimizes the pose graph and moves
  // the map. Returns true if the map was corrected.
  bool process(int id, LocalMap& map);

  const std::vector<LoopEdge>& loops() const { return loops_; }

 private:
  std::vector<std::pair<int, double>> query(const BowVector& v, int id,
                                            double min_score) const;
  bool verify(int cand, int cur, const LocalMap& map, LoopEdge& edge);
  void correct(LocalMap& map) const;

  Vocabulary voc_;
  Camera cam_;
  LoopParams params_;
  std::mt19937 rng_{11};
  std::vector<BowVector> bow_;                         // per keyframe
  std::unordered_map<int, std::vector<int>> inverted_;  // word -> keyframes
  std::vector<LoopEdge> loops_;
};

}  // namespace lynx
