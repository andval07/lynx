#include "lynx/odometry.hpp"

#include <chrono>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace lynx {

RgbdOdometry::RgbdOdometry(const TumCalibration& calib, OdometryParams p)
    : calib_(calib), params_(p), frontend_(calib), map_(calib.cam, p.ba.bf) {}

std::vector<double> RgbdOdometry::feature_depths(const Features& f,
                                                 const cv::Mat& depth) const {
  std::vector<double> d(f.keypoints.size(), -1.0);
  for (size_t i = 0; i < f.keypoints.size(); ++i) {
    // Depth is registered to the raw (distorted) image, so look it up there.
    const auto& pt = f.keypoints[i].pt;
    const int u = cvRound(pt.x), v = cvRound(pt.y);
    if (u < 0 || v < 0 || u >= depth.cols || v >= depth.rows) continue;
    const double z = depth.at<uint16_t>(v, u) / calib_.depth_scale;
    if (z >= params_.min_depth && z <= params_.max_depth) d[i] = z;
  }
  return d;
}

bool RgbdOdometry::needs_keyframe(const SE3& T_wc, int inliers) const {
  if (inliers < params_.kf_min_inliers) return true;
  const SE3 rel = map_.keyframe(map_.n_keyframes() - 1).T_wc.inverse() * T_wc;
  const double angle_deg = so3_log(rel.R()).norm() * 180.0 / M_PI;
  return rel.t().norm() > params_.kf_max_translation ||
         angle_deg > params_.kf_max_rotation_deg;
}

FrameResult RgbdOdometry::track(const cv::Mat& gray, const cv::Mat& depth) {
  const Features f = frontend_.extract(gray);
  FrameResult r;
  if (map_.n_keyframes() == 0) {
    map_.add_keyframe(SE3(), f, feature_depths(f, depth), {});
    frames_.push_back({0, SE3()});
    r.tracked = r.keyframe = true;
    return r;
  }

  const int kf_id = map_.n_keyframes() - 1;
  const MapKeyframe& kf = map_.keyframe(kf_id);
  const auto matches = OrbFrontend::match(f.descriptors, kf.descriptors,
                                          params_.ratio);
  std::vector<Vec3> pw;
  std::vector<Vec2> uv;
  std::vector<std::pair<int, int>> feature_point;
  for (const auto& m : matches) {
    const int pid = kf.point[m.trainIdx];
    if (pid < 0) continue;
    pw.push_back(map_.point(pid).p_w);
    uv.push_back(f.uv[m.queryIdx]);
    feature_point.emplace_back(m.queryIdx, pid);
  }
  r.matches = static_cast<int>(pw.size());

  const PnPResult pnp = pnp_ransac(pw, uv, calib_.cam, params_.ransac, rng_);
  std::vector<std::pair<int, int>> inlier_pairs;
  if (pnp.ok) {
    r.T_wc = pnp.T_cw.inverse();
    r.inliers = static_cast<int>(pnp.inliers.size());
    r.tracked = true;
    for (int i : pnp.inliers) inlier_pairs.push_back(feature_point[i]);
  } else {
    // Lost: keep moving with the last motion and restart from this frame.
    r.T_wc = T_wc_last_ * motion_;
  }

  if (!r.tracked || needs_keyframe(r.T_wc, r.inliers)) {
    const int id = map_.add_keyframe(r.T_wc, f, feature_depths(f, depth),
                                     inlier_pairs);
    if (params_.local_ba) {
      map_.local_bundle_adjust(params_.ba_window, params_.ba);
      r.T_wc = map_.keyframe(id).T_wc;
    }
    frames_.push_back({id, SE3()});
    r.keyframe = true;
  } else {
    frames_.push_back({kf_id, kf.T_wc.inverse() * r.T_wc});
  }
  motion_ = T_wc_last_.inverse() * r.T_wc;
  T_wc_last_ = r.T_wc;
  return r;
}

std::vector<SE3> RgbdOdometry::refined_poses() const {
  std::vector<SE3> out;
  out.reserve(frames_.size());
  for (const auto& fr : frames_) {
    out.push_back(map_.keyframe(fr.kf).T_wc * fr.T_kf_c);
  }
  return out;
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
  // Report the poses after the last bundle adjustment.
  const std::vector<SE3> refined = vo.refined_poses();
  for (size_t i = 0; i < s.trajectory.size(); ++i) s.trajectory[i].T_wc = refined[i];
  s.keyframes = vo.keyframes();
  s.ms_per_frame = s.frames > 0 ? total_ms / s.frames : 0;
  return s;
}

}  // namespace lynx
