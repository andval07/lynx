#pragma once

#include <random>
#include <string>

#include "lynx/frontend.hpp"
#include "lynx/geometry.hpp"

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
};

struct FrameResult {
  SE3 T_wc;
  int matches = 0;
  int inliers = 0;
  bool tracked = false;
  bool keyframe = false;
};

// RGB-D visual odometry, frame to keyframe: the keyframe's ORB points are
// lifted to 3D with depth, and each new frame is located by PnP against them.
class RgbdOdometry {
 public:
  explicit RgbdOdometry(const TumCalibration& calib, OdometryParams p = {});

  // gray: 8-bit image, depth: 16-bit TUM depth image.
  FrameResult track(const cv::Mat& gray, const cv::Mat& depth);

  int keyframes() const { return n_keyframes_; }

 private:
  struct Keyframe {
    SE3 T_wc;
    std::vector<Vec3> points_w;
    cv::Mat descriptors;
  };

  Keyframe make_keyframe(const Features& f, const cv::Mat& depth,
                         const SE3& T_wc) const;
  bool needs_keyframe(const SE3& T_wc, int inliers) const;

  TumCalibration calib_;
  OdometryParams params_;
  OrbFrontend frontend_;
  std::mt19937 rng_{42};

  bool initialized_ = false;
  Keyframe kf_;
  int n_keyframes_ = 0;
  SE3 T_wc_last_;
  SE3 motion_;  // T_prev_last: last frame-to-frame motion, for prediction
};

struct RunStats {
  Trajectory trajectory;
  std::vector<FrameResult> per_frame;
  int frames = 0;
  int keyframes = 0;
  int lost = 0;
  double ms_per_frame = 0;
};

// Runs the odometry on a TUM sequence folder. max_frames <= 0 means all.
RunStats run_tum_sequence(const std::string& dir, int max_frames = 0,
                          OdometryParams p = {});

}  // namespace lynx
