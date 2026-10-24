#include "lynx/odometry.hpp"

#include <chrono>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace lynx {

RgbdOdometry::RgbdOdometry(const TumCalibration& calib, OdometryParams p)
    : calib_(calib), params_(p), frontend_(calib) {}

RgbdOdometry::Keyframe RgbdOdometry::make_keyframe(const Features& f,
                                                   const cv::Mat& depth,
                                                   const SE3& T_wc) const {
  Keyframe kf;
  kf.T_wc = T_wc;
  std::vector<int> rows;
  for (size_t i = 0; i < f.keypoints.size(); ++i) {
    // Depth is registered to the raw (distorted) image, so look it up there.
    const auto& pt = f.keypoints[i].pt;
    const int u = cvRound(pt.x), v = cvRound(pt.y);
    if (u < 0 || v < 0 || u >= depth.cols || v >= depth.rows) continue;
    const double d = depth.at<uint16_t>(v, u) / calib_.depth_scale;
    if (d < params_.min_depth || d > params_.max_depth) continue;
    kf.points_w.push_back(T_wc * calib_.cam.backproject(f.uv[i], d));
    rows.push_back(static_cast<int>(i));
  }
  kf.descriptors = cv::Mat(static_cast<int>(rows.size()), f.descriptors.cols,
                           f.descriptors.type());
  for (size_t k = 0; k < rows.size(); ++k) {
    f.descriptors.row(rows[k]).copyTo(kf.descriptors.row(static_cast<int>(k)));
  }
  return kf;
}

bool RgbdOdometry::needs_keyframe(const SE3& T_wc, int inliers) const {
  if (inliers < params_.kf_min_inliers) return true;
  const SE3 rel = kf_.T_wc.inverse() * T_wc;
  const double angle_deg = so3_log(rel.R()).norm() * 180.0 / M_PI;
  return rel.t().norm() > params_.kf_max_translation ||
         angle_deg > params_.kf_max_rotation_deg;
}

FrameResult RgbdOdometry::track(const cv::Mat& gray, const cv::Mat& depth) {
  const Features f = frontend_.extract(gray);
  FrameResult r;
  if (!initialized_) {
    kf_ = make_keyframe(f, depth, SE3());
    initialized_ = true;
    ++n_keyframes_;
    r.tracked = r.keyframe = true;
    return r;
  }

  const auto matches = OrbFrontend::match(f.descriptors, kf_.descriptors,
                                          params_.ratio);
  std::vector<Vec3> pw;
  std::vector<Vec2> uv;
  pw.reserve(matches.size());
  uv.reserve(matches.size());
  for (const auto& m : matches) {
    pw.push_back(kf_.points_w[m.trainIdx]);
    uv.push_back(f.uv[m.queryIdx]);
  }
  r.matches = static_cast<int>(matches.size());

  const PnPResult pnp = pnp_ransac(pw, uv, calib_.cam, params_.ransac, rng_);
  if (pnp.ok) {
    r.T_wc = pnp.T_cw.inverse();
    r.inliers = static_cast<int>(pnp.inliers.size());
    r.tracked = true;
  } else {
    // Lost: keep moving with the last motion and restart from this frame.
    r.T_wc = T_wc_last_ * motion_;
  }

  if (!r.tracked || needs_keyframe(r.T_wc, r.inliers)) {
    kf_ = make_keyframe(f, depth, r.T_wc);
    ++n_keyframes_;
    r.keyframe = true;
  }
  motion_ = T_wc_last_.inverse() * r.T_wc;
  T_wc_last_ = r.T_wc;
  return r;
}

RunStats run_tum_sequence(const std::string& dir, int max_frames,
                          OdometryParams p) {
  const auto frames = load_tum_sequence(dir);
  RgbdOdometry vo(tum_calibration(dir), p);
  RunStats s;
  double total_ms = 0;
  for (const auto& fr : frames) {
    if (max_frames > 0 && s.frames >= max_frames) break;
    const cv::Mat gray = cv::imread(dir + "/" + fr.rgb, cv::IMREAD_GRAYSCALE);
    const cv::Mat depth = cv::imread(dir + "/" + fr.depth, cv::IMREAD_ANYDEPTH);
    if (gray.empty() || depth.empty() || depth.type() != CV_16U) {
      throw std::runtime_error("cannot read frame " + fr.rgb);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const FrameResult r = vo.track(gray, depth);
    const auto t1 = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.trajectory.push_back({fr.stamp, r.T_wc});
    s.per_frame.push_back(r);
    if (!r.tracked) ++s.lost;
    ++s.frames;
  }
  s.keyframes = vo.keyframes();
  s.ms_per_frame = s.frames > 0 ? total_ms / s.frames : 0;
  return s;
}

}  // namespace lynx
