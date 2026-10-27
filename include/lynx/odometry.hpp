#pragma once

#include <optional>
#include <random>
#include <string>
#include <vector>

#include "lynx/frontend.hpp"
#include "lynx/geometry.hpp"
#include "lynx/loop_closure.hpp"
#include "lynx/map.hpp"

namespace lynx {

struct OdometryParams {
  RansacParams ransac;
  float ratio = 0.8f;
  double min_depth = 0.1;  // meters
  double max_depth = 5.0;
  // A new keyframe starts when tracking gets weak or the camera moved far.
  int kf_min_inliers = 120;
  double kf_max_translation = 0.25;  // meters
  double kf_max_rotation_deg = 15.0;
  // Local bundle adjustment after each new keyframe.
  bool local_ba = true;
  int ba_window = 7;  // keyframes
  BaOptions ba;
  // Loop closure runs when a vocabulary file is given.
  std::string vocabulary;
  LoopParams loop;
};

struct FrameResult {
  SE3 T_wc;
  int matches = 0;
  int inliers = 0;
  bool tracked = false;
  bool keyframe = false;
  bool loop = false;  // a loop closure corrected the map at this frame
};

// RGB-D visual odometry, frame to keyframe: each new frame is located by PnP
// against the map points seen by the last keyframe. With local_ba on, every
// new keyframe triggers a bundle adjustment of the recent keyframes.
class RgbdOdometry {
 public:
  explicit RgbdOdometry(const TumCalibration& calib, OdometryParams p = {});

  // gray: 8-bit image, depth: 16-bit TUM depth image.
  FrameResult track(const cv::Mat& gray, const cv::Mat& depth);

  int keyframes() const { return map_.n_keyframes(); }
  std::vector<LoopEdge> loops() const;

  // Pose of every tracked frame, recomputed from its reference keyframe so it
  // includes the corrections bundle adjustment made after the frame was seen.
  std::vector<SE3> refined_poses() const;

 private:
  struct FrameRef {
    int kf;
    SE3 T_kf_c;
  };

  std::vector<double> feature_depths(const Features& f, const cv::Mat& depth) const;
  bool needs_keyframe(const SE3& T_wc, int inliers) const;

  TumCalibration calib_;
  OdometryParams params_;
  OrbFrontend frontend_;
  LocalMap map_;
  std::optional<LoopCloser> loop_;
  std::mt19937 rng_{42};

  std::vector<FrameRef> frames_;
  SE3 T_wc_last_;
  SE3 motion_;  // T_prev_last: last frame-to-frame motion, for prediction
};

struct RunStats {
  Trajectory trajectory;
  std::vector<FrameResult> per_frame;
  int frames = 0;
  int keyframes = 0;
  int lost = 0;
  std::vector<LoopEdge> loops;
  double ms_per_frame = 0;
};

// Runs the odometry on a TUM sequence folder. max_frames <= 0 means all.
RunStats run_tum_sequence(const std::string& dir, int max_frames = 0,
                          OdometryParams p = {});

}  // namespace lynx
