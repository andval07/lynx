#include "lynx/mono_odometry.hpp"

#include <algorithm>
#include <cmath>

namespace lynx {

namespace {

BaOptions mono_ba(BaOptions o) {
  o.bf = 0;
  return o;
}

}  // namespace

MonoOdometry::MonoOdometry(const TumCalibration& calib, MonoParams p)
    : cam_(calib.cam), params_(p), frontend_(calib), map_(calib.cam, 0.0) {
  params_.ba = mono_ba(p.ba);
}

FrameResult MonoOdometry::track(const cv::Mat& gray) {
  return track(frontend_.extract(gray));
}

FrameResult MonoOdometry::try_initialize(const Features& f) {
  FrameResult r;
  if (ref_.uv.empty()) {
    ref_ = f;
    ref_frame_ = frames_.size();
    frames_.push_back({-1, SE3()});
    return r;
  }
  const auto matches = OrbFrontend::match(f.descriptors, ref_.descriptors,
                                          params_.ratio);
  if (static_cast<int>(matches.size()) < params_.init_min_matches) {
    ref_ = f;
    ref_frame_ = frames_.size();
    frames_.push_back({-1, SE3()});
    return r;
  }
  std::vector<Vec2> uv1, uv2;
  for (const auto& m : matches) {
    uv1.push_back(ref_.uv[m.trainIdx]);
    uv2.push_back(f.uv[m.queryIdx]);
  }
  const TwoViewResult tv = initialize_two_view(uv1, uv2, cam_, params_.init, rng_);
  if (!tv.ok) {
    frames_.push_back({-1, SE3()});
    return r;
  }

  // Scale the map so that the median depth in the first camera is 1.
  std::vector<double> z;
  for (size_t i = 0; i < matches.size(); ++i) {
    if (tv.good[i]) z.push_back(tv.points[i].z());
  }
  std::nth_element(z.begin(), z.begin() + z.size() / 2, z.end());
  const double s = 1.0 / z[z.size() / 2];

  std::vector<double> depth0(ref_.uv.size(), -1.0);
  for (size_t i = 0; i < matches.size(); ++i) {
    if (tv.good[i]) depth0[matches[i].trainIdx] = tv.points[i].z() * s;
  }
  map_.add_keyframe(SE3(), ref_, depth0, {});
  std::vector<std::pair<int, int>> matched;
  for (size_t i = 0; i < matches.size(); ++i) {
    const int pid = map_.keyframe(0).point[matches[i].trainIdx];
    if (tv.good[i] && pid >= 0) matched.emplace_back(matches[i].queryIdx, pid);
  }
  const SE3 T_21(tv.T_21.R(), tv.T_21.t() * s);
  map_.add_keyframe(T_21.inverse(), f, std::vector<double>(f.uv.size(), -1.0),
                    matched);
  if (params_.local_ba) map_.local_bundle_adjust(params_.ba_window, params_.ba);

  frames_[ref_frame_] = {0, SE3()};
  frames_.push_back({1, SE3()});
  r.T_wc = map_.keyframe(1).T_wc;
  r.matches = static_cast<int>(matches.size());
  r.inliers = static_cast<int>(matched.size());
  r.tracked = r.keyframe = true;
  T_wc_last_ = r.T_wc;
  motion_ = map_.keyframe(0).T_wc.inverse() * r.T_wc;
  return r;
}

std::vector<double> MonoOdometry::triangulate_with(
    int kf_id, const Features& f, const SE3& T_wc,
    const std::vector<bool>& has_point,
    std::vector<std::pair<int, int>>& pairs) const {
  const MapKeyframe& kf = map_.keyframe(kf_id);
  const SE3 T1 = kf.T_wc.inverse(), T2 = T_wc.inverse();
  const Mat34 P1 = projection_matrix(cam_, T1), P2 = projection_matrix(cam_, T2);
  const double min_cos = std::cos(params_.min_parallax_deg * M_PI / 180.0);

  std::vector<double> depth(f.uv.size(), -1.0);
  for (const auto& m : OrbFrontend::match(f.descriptors, kf.descriptors,
                                          params_.ratio)) {
    const int q = m.queryIdx, t = m.trainIdx;
    if (has_point[q] || kf.point[t] >= 0) continue;
    const Vec3 X = triangulate(P1, P2, kf.uv[t], f.uv[q]);
    const Vec3 p1 = T1 * X, p2 = T2 * X;
    if (p1.z() <= 0 || p2.z() <= 0) continue;
    if ((cam_.project(p1) - kf.uv[t]).norm() > params_.max_reproj_px ||
        (cam_.project(p2) - f.uv[q]).norm() > params_.max_reproj_px) {
      continue;
    }
    const Vec3 a = X - kf.T_wc.t(), b = X - T_wc.t();
    if (a.normalized().dot(b.normalized()) > min_cos) continue;
    depth[q] = p2.z();
    pairs.emplace_back(q, t);
  }
  return depth;
}

int MonoOdometry::add_keyframe(const SE3& T_wc, const Features& f,
                               const std::vector<std::pair<int, int>>& matched) {
  const int prev = map_.n_keyframes() - 1;
  std::vector<bool> has_point(f.uv.size(), false);
  for (const auto& fp : matched) has_point[fp.first] = true;
  std::vector<std::pair<int, int>> pairs;
  const auto depth = triangulate_with(prev, f, T_wc, has_point, pairs);
  const int id = map_.add_keyframe(T_wc, f, depth, matched);
  for (const auto& [q, t] : pairs) {
    const int pid = map_.keyframe(id).point[q];
    if (pid >= 0) map_.add_observation(pid, prev, t);
  }
  if (params_.local_ba) map_.local_bundle_adjust(params_.ba_window, params_.ba);
  return id;
}

bool MonoOdometry::needs_keyframe(const SE3& T_wc, int inliers) const {
  if (inliers < params_.kf_min_inliers) return true;
  const SE3 rel = map_.keyframe(map_.n_keyframes() - 1).T_wc.inverse() * T_wc;
  const double angle_deg = so3_log(rel.R()).norm() * 180.0 / M_PI;
  return rel.t().norm() > params_.kf_max_translation ||
         angle_deg > params_.kf_max_rotation_deg;
}

FrameResult MonoOdometry::track(const Features& f) {
  if (!initialized()) return try_initialize(f);

  FrameResult r;
  const int kf_id = map_.n_keyframes() - 1;
  const MapKeyframe& kf = map_.keyframe(kf_id);
  std::vector<Vec3> pw;
  std::vector<Vec2> uv;
  std::vector<std::pair<int, int>> feature_point;
  for (const auto& m : OrbFrontend::match(f.descriptors, kf.descriptors,
                                          params_.ratio)) {
    const int pid = kf.point[m.trainIdx];
    if (pid < 0) continue;
    pw.push_back(map_.point(pid).p_w);
    uv.push_back(f.uv[m.queryIdx]);
    feature_point.emplace_back(m.queryIdx, pid);
  }
  r.matches = static_cast<int>(pw.size());

  const PnPResult pnp = pnp_ransac(pw, uv, cam_, params_.ransac, rng_);
  if (!pnp.ok) {
    // Without depth a keyframe here would have no points: only predict.
    r.T_wc = T_wc_last_ * motion_;
    frames_.push_back({kf_id, kf.T_wc.inverse() * r.T_wc});
  } else {
    r.T_wc = pnp.T_cw.inverse();
    r.inliers = static_cast<int>(pnp.inliers.size());
    r.tracked = true;
    if (needs_keyframe(r.T_wc, r.inliers)) {
      std::vector<std::pair<int, int>> inlier_pairs;
      for (int i : pnp.inliers) inlier_pairs.push_back(feature_point[i]);
      const int id = add_keyframe(r.T_wc, f, inlier_pairs);
      r.T_wc = map_.keyframe(id).T_wc;
      frames_.push_back({id, SE3()});
      r.keyframe = true;
    } else {
      frames_.push_back({kf_id, kf.T_wc.inverse() * r.T_wc});
    }
  }
  motion_ = T_wc_last_.inverse() * r.T_wc;
  T_wc_last_ = r.T_wc;
  return r;
}

std::vector<SE3> MonoOdometry::refined_poses() const {
  std::vector<SE3> out;
  out.reserve(frames_.size());
  for (const auto& fr : frames_) {
    out.push_back(fr.kf < 0 ? SE3() : map_.keyframe(fr.kf).T_wc * fr.T_kf_c);
  }
  return out;
}

}  // namespace lynx
