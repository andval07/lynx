#pragma once

#include <random>
#include <vector>

#include "lynx/frontend.hpp"
#include "lynx/map.hpp"
#include "lynx/odometry.hpp"
#include "lynx/two_view.hpp"

namespace lynx {

struct MonoParams {
  RansacParams ransac;
  float ratio = 0.8f;
  TwoViewParams init;
  int init_min_matches = 100;  // below this the reference frame is replaced
  // The map is scaled so that the median depth at initialization is 1, and
  // the translation threshold is in these units.
  int kf_min_inliers = 100;
  double kf_max_translation = 0.1;
  double kf_max_rotation_deg = 10.0;
  // New points from two keyframes.
  double min_parallax_deg = 1.0;
  double max_reproj_px = 4.0;
  int triangulate_window = 4;  // keyframes tried for new points
  int track_window = 3;        // keyframes whose points each frame is matched to
  bool local_ba = true;
  int ba_window = 7;
  BaOptions ba;  // bf is ignored: monocular residuals are 2D only
};

// Monocular visual odometry. Two-view initialization gives the first two
// keyframes and the scale; then each frame is located by PnP against the
// points of the last keyframe, and every new keyframe triangulates new points
// with the keyframe before it. The scale is arbitrary and can drift.
class MonoOdometry {
 public:
  MonoOdometry(const TumCalibration& calib, MonoParams p = {});

  FrameResult track(const cv::Mat& gray);
  // Same, on features that were already extracted (used by the tests).
  FrameResult track(const Features& f);

  bool initialized() const { return map_.n_keyframes() > 0; }
  int keyframes() const { return map_.n_keyframes(); }
  const LocalMap& map() const { return map_; }

  // Pose of every frame from its reference keyframe; frames before the
  // initialization keep the identity and are marked as not tracked.
  std::vector<SE3> refined_poses() const;
  // False for the frames before the initialization, which have no pose.
  bool has_pose(size_t frame) const { return frames_[frame].kf >= 0; }

 private:
  struct FrameRef {
    int kf;
    SE3 T_kf_c;
  };

  FrameResult try_initialize(const Features& f);
  std::vector<double> triangulate_with(int kf_id, const Features& f,
                                       const SE3& T_wc,
                                       const std::vector<bool>& has_point,
                                       std::vector<std::pair<int, int>>& pairs) const;
  int add_keyframe(const SE3& T_wc, const Features& f,
                   const std::vector<std::pair<int, int>>& matched);
  bool needs_keyframe(const SE3& T_wc, int inliers) const;

  Camera cam_;
  MonoParams params_;
  OrbFrontend frontend_;
  LocalMap map_;
  std::mt19937 rng_{42};

  Features ref_;      // first frame of the initialization pair
  size_t ref_frame_ = 0;
  std::vector<FrameRef> frames_;
  SE3 T_wc_last_;
  SE3 motion_;
};

}  // namespace lynx
