#include "lynx/map.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>

namespace lynx {

namespace {
constexpr double kOrbScale = 1.2;  // same as the frontend's ORB pyramid
}  // namespace

int LocalMap::add_keyframe(const SE3& T_wc, const Features& f,
                           const std::vector<double>& depth,
                           const std::vector<std::pair<int, int>>& matched) {
  const int id = n_keyframes();
  MapKeyframe kf;
  kf.T_wc = T_wc;
  kf.uv = f.uv;
  kf.descriptors = f.descriptors.clone();
  kf.point.assign(f.uv.size(), -1);
  kf.u_right.assign(f.uv.size(), -1.0);
  kf.sigma.resize(f.uv.size());
  for (size_t i = 0; i < f.uv.size(); ++i) {
    if (depth[i] > 0 && bf_ > 0) kf.u_right[i] = f.uv[i].x() - bf_ / depth[i];
    kf.sigma[i] = std::pow(kOrbScale, f.keypoints[i].octave);
  }

  std::vector<bool> seen(points_.size(), false);
  for (const auto& [feature, pid] : matched) {
    if (seen[pid] || points_[pid].obs.empty()) continue;
    seen[pid] = true;
    kf.point[feature] = pid;
    points_[pid].obs.emplace_back(id, feature);
  }
  for (size_t i = 0; i < f.uv.size(); ++i) {
    if (kf.point[i] >= 0 || depth[i] <= 0) continue;
    MapPoint p;
    p.p_w = T_wc * cam_.backproject(f.uv[i], depth[i]);
    p.obs.emplace_back(id, static_cast<int>(i));
    kf.point[i] = static_cast<int>(points_.size());
    points_.push_back(std::move(p));
  }
  kfs_.push_back(std::move(kf));
  return id;
}

void LocalMap::add_observation(int point_id, int kf, int feature) {
  if (kfs_[kf].point[feature] >= 0 || points_[point_id].obs.empty()) return;
  kfs_[kf].point[feature] = point_id;
  points_[point_id].obs.emplace_back(kf, feature);
}

void LocalMap::remove_observation(int point_id, int kf, int feature) {
  auto& obs = points_[point_id].obs;
  obs.erase(std::remove(obs.begin(), obs.end(), std::make_pair(kf, feature)),
            obs.end());
  kfs_[kf].point[feature] = -1;
}

LocalBaStats LocalMap::local_bundle_adjust(int window, const BaOptions& opt) {
  LocalBaStats st;
  const int n = n_keyframes();
  if (n < 2) return st;
  const int first = std::max(0, n - window);
  // Without depth the scale is free: the two oldest keyframes of the window
  // stay fixed, so that it cannot shrink to zero.
  const int first_free = bf_ > 0 || n <= 2 ? first : std::min(first + 2, n - 1);

  // Problem indices for keyframes and points, filled on first use.
  std::unordered_map<int, int> pose_of_kf, var_of_point;
  std::vector<int> kf_of_pose, point_of_var;
  BaProblem prob;
  auto pose_index = [&](int kf) {
    auto [it, inserted] = pose_of_kf.try_emplace(kf, static_cast<int>(prob.T_cw.size()));
    if (inserted) {
      prob.T_cw.push_back(kfs_[kf].T_wc.inverse());
      prob.fixed.push_back(kf < first_free || kf == 0);
      kf_of_pose.push_back(kf);
    }
    return it->second;
  };

  std::vector<std::pair<int, int>> obs_ref;  // (kf, feature) per observation
  for (int k = first; k < n; ++k) {
    for (int pid : kfs_[k].point) {
      if (pid < 0 || var_of_point.count(pid)) continue;
      var_of_point[pid] = static_cast<int>(prob.points.size());
      prob.points.push_back(points_[pid].p_w);
      point_of_var.push_back(pid);
      for (const auto& [kf, feature] : points_[pid].obs) {
        BaObservation o;
        o.pose = pose_index(kf);
        o.point = var_of_point[pid];
        o.uv = kfs_[kf].uv[feature];
        o.u_right = kfs_[kf].u_right[feature];
        o.sigma = kfs_[kf].sigma[feature];
        prob.obs.push_back(o);
        obs_ref.emplace_back(kf, feature);
      }
    }
  }

  const BaResult r = bundle_adjust(prob, cam_, opt);

  for (size_t i = 0; i < prob.T_cw.size(); ++i) {
    if (prob.fixed[i]) {
      ++st.fixed_poses;
    } else {
      kfs_[kf_of_pose[i]].T_wc = r.T_cw[i].inverse();
      ++st.free_poses;
    }
  }
  for (size_t j = 0; j < r.points.size(); ++j) {
    points_[point_of_var[j]].p_w = r.points[j];
  }
  for (size_t k = 0; k < prob.obs.size(); ++k) {
    if (!r.outlier[k]) continue;
    remove_observation(point_of_var[prob.obs[k].point], obs_ref[k].first,
                       obs_ref[k].second);
    ++st.outliers;
  }
  st.points = static_cast<int>(prob.points.size());
  st.observations = static_cast<int>(prob.obs.size());
  return st;
}

std::vector<std::array<int, 3>> LocalMap::covisibility(int min_shared) const {
  std::map<std::pair<int, int>, int> shared;
  for (const MapPoint& p : points_) {
    for (size_t a = 0; a < p.obs.size(); ++a) {
      for (size_t b = a + 1; b < p.obs.size(); ++b) {
        const int i = std::min(p.obs[a].first, p.obs[b].first);
        const int j = std::max(p.obs[a].first, p.obs[b].first);
        if (i != j) ++shared[{i, j}];
      }
    }
  }
  std::vector<std::array<int, 3>> out;
  for (const auto& [ij, n] : shared) {
    if (n >= min_shared) out.push_back({ij.first, ij.second, n});
  }
  return out;
}

void LocalMap::set_keyframe_poses(const std::vector<SE3>& T_wc) {
  for (MapPoint& p : points_) {
    if (p.obs.empty()) continue;
    const int k = p.obs.back().first;
    p.p_w = T_wc[k] * (kfs_[k].T_wc.inverse() * p.p_w);
  }
  for (size_t k = 0; k < kfs_.size(); ++k) kfs_[k].T_wc = T_wc[k];
}

}  // namespace lynx
