#pragma once

#include <opencv2/core.hpp>
#include <utility>
#include <vector>

#include "lynx/ba.hpp"
#include "lynx/frontend.hpp"

namespace lynx {

struct MapKeyframe {
  SE3 T_wc;
  std::vector<Vec2> uv;         // undistorted pixels, one per feature
  std::vector<double> u_right;  // virtual right coordinate, < 0 without depth
  std::vector<double> sigma;    // pixel noise, 1.2^octave
  cv::Mat descriptors;
  std::vector<int> point;       // map point id per feature, -1 for none
};

struct MapPoint {
  Vec3 p_w;
  std::vector<std::pair<int, int>> obs;  // (keyframe, feature); empty = removed
};

struct LocalBaStats {
  int free_poses = 0;
  int fixed_poses = 0;
  int points = 0;
  int observations = 0;
  int outliers = 0;
};

// Keyframes and the 3D points they share. Points are created from depth and
// get more observations when later keyframes match them while tracking.
class LocalMap {
 public:
  LocalMap(const Camera& cam, double bf) : cam_(cam), bf_(bf) {}

  // matched: (feature index, map point id) pairs found by tracking. Every
  // other feature with depth (> 0) starts a new point. Returns the index.
  int add_keyframe(const SE3& T_wc, const Features& f,
                   const std::vector<double>& depth,
                   const std::vector<std::pair<int, int>>& matched);

  // Bundle adjustment of the last `window` keyframes and the points they see.
  // Other keyframes that see those points enter as fixed poses; keyframe 0
  // is always fixed. Outlier observations are dropped from the map.
  LocalBaStats local_bundle_adjust(int window, const BaOptions& opt);

  int n_keyframes() const { return static_cast<int>(kfs_.size()); }
  const MapKeyframe& keyframe(int i) const { return kfs_[i]; }
  const MapPoint& point(int id) const { return points_[id]; }

 private:
  void remove_observation(int point_id, int kf, int feature);

  Camera cam_;
  double bf_;
  std::vector<MapKeyframe> kfs_;
  std::vector<MapPoint> points_;
};

}  // namespace lynx
