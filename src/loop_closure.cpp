#include "lynx/loop_closure.hpp"

#include <algorithm>
#include <utility>

namespace lynx {

LoopCloser::LoopCloser(Vocabulary voc, const Camera& cam, LoopParams p)
    : voc_(std::move(voc)), cam_(cam), params_(p) {}

std::vector<std::pair<int, double>> LoopCloser::query(const BowVector& v, int id,
                                                      double min_score) const {
  // Only keyframes that share a word can score above zero.
  std::vector<bool> seen(bow_.size(), false);
  std::vector<std::pair<int, double>> out;
  for (const auto& [w, x] : v) {
    const auto it = inverted_.find(w);
    if (it == inverted_.end()) continue;
    for (int k : it->second) {
      if (seen[k] || k > id - params_.min_gap) continue;
      seen[k] = true;
      const double s = bow_score(v, bow_[k]);
      if (s >= min_score) out.emplace_back(k, s);
    }
  }
  std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  if (static_cast<int>(out.size()) > params_.max_candidates) {
    out.resize(params_.max_candidates);
  }
  return out;
}

bool LoopCloser::verify(int cand, int cur, const LocalMap& map, LoopEdge& edge) {
  const MapKeyframe& kc = map.keyframe(cand);
  const MapKeyframe& kf = map.keyframe(cur);
  const auto matches = OrbFrontend::match(kf.descriptors, kc.descriptors, params_.ratio);
  // Points in the candidate camera frame, so PnP gives T_cur_cand directly.
  const SE3 T_cw_cand = kc.T_wc.inverse();
  std::vector<Vec3> pc;
  std::vector<Vec2> uv;
  for (const auto& m : matches) {
    const int pid = kc.point[m.trainIdx];
    if (pid < 0 || map.point(pid).obs.empty()) continue;
    pc.push_back(T_cw_cand * map.point(pid).p_w);
    uv.push_back(kf.uv[m.queryIdx]);
  }
  if (static_cast<int>(pc.size()) < params_.min_inliers) return false;
  const PnPResult pnp = pnp_ransac(pc, uv, cam_, params_.ransac, rng_);
  if (!pnp.ok || static_cast<int>(pnp.inliers.size()) < params_.min_inliers) return false;
  edge.cand = cand;
  edge.cur = cur;
  edge.T_cand_cur = pnp.T_cw.inverse();
  edge.inliers = static_cast<int>(pnp.inliers.size());
  return true;
}

void LoopCloser::correct(LocalMap& map) const {
  // Odometry and covisibility edges keep the relative poses the map has now;
  // the loop edges pull the chain together.
  const int n = map.n_keyframes();
  PoseGraph g;
  for (int k = 0; k < n; ++k) g.T_w.push_back(map.keyframe(k).T_wc);
  g.fixed.assign(n, false);
  g.fixed[0] = true;
  const auto rel = [&](int i, int j) { return g.T_w[i].inverse() * g.T_w[j]; };
  for (int k = 1; k < n; ++k) g.edges.push_back({k - 1, k, rel(k - 1, k)});
  for (const auto& c : map.covisibility(params_.covis_min_shared)) {
    if (c[1] - c[0] > 1) g.edges.push_back({c[0], c[1], rel(c[0], c[1])});
  }
  for (const LoopEdge& l : loops_) g.edges.push_back({l.cand, l.cur, l.T_cand_cur});
  map.set_keyframe_poses(optimize_pose_graph(g, params_.pgo).T_w);
}

bool LoopCloser::process(int id, LocalMap& map) {
  const BowVector v = voc_.transform(map.keyframe(id).descriptors);
  bool closed = false;
  if (id > 0 && id >= params_.min_gap) {
    const double s_prev = bow_score(v, bow_[id - 1]);
    const double min_score = std::max(params_.min_score, params_.min_score_ratio * s_prev);
    for (const auto& [cand, score] : query(v, id, min_score)) {
      LoopEdge e;
      if (!verify(cand, id, map, e)) continue;
      e.score = score;
      loops_.push_back(e);
      correct(map);
      closed = true;
      break;
    }
  }
  bow_.push_back(v);
  for (const auto& [w, x] : v) inverted_[w].push_back(id);
  return closed;
}

}  // namespace lynx
