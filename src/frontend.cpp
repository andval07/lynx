#include "lynx/frontend.hpp"

#include <opencv2/calib3d.hpp>
#include <unordered_map>

namespace lynx {

OrbFrontend::OrbFrontend(const TumCalibration& calib, int n_features)
    : orb_(cv::ORB::create(n_features, 1.2f, 8)),
      K_((cv::Mat_<double>(3, 3) << calib.cam.fx, 0, calib.cam.cx, 0,
          calib.cam.fy, calib.cam.cy, 0, 0, 1)),
      dist_(cv::Mat(calib.dist.size(), 1, CV_64F,
                    const_cast<double*>(calib.dist.data()))
                .clone()) {}

Features OrbFrontend::extract(const cv::Mat& gray) const {
  Features f;
  orb_->detectAndCompute(gray, cv::noArray(), f.keypoints, f.descriptors);
  if (f.keypoints.empty()) return f;

  std::vector<cv::Point2f> raw, und;
  raw.reserve(f.keypoints.size());
  for (const auto& kp : f.keypoints) raw.push_back(kp.pt);
  // P = K maps the undistorted normalized points back to pixels.
  cv::undistortPoints(raw, und, K_, dist_, cv::noArray(), K_);
  f.uv.reserve(und.size());
  for (const auto& p : und) f.uv.emplace_back(p.x, p.y);
  return f;
}

std::vector<cv::DMatch> OrbFrontend::match(const cv::Mat& query,
                                           const cv::Mat& train, float ratio) {
  std::vector<cv::DMatch> out;
  if (query.rows < 2 || train.rows < 2) return out;
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(query, train, knn, 2);

  std::unordered_map<int, cv::DMatch> best_for_train;
  for (const auto& m : knn) {
    if (m.size() < 2 || m[0].distance >= ratio * m[1].distance) continue;
    auto it = best_for_train.find(m[0].trainIdx);
    if (it == best_for_train.end() || m[0].distance < it->second.distance) {
      best_for_train[m[0].trainIdx] = m[0];
    }
  }
  out.reserve(best_for_train.size());
  for (const auto& [idx, m] : best_for_train) out.push_back(m);
  return out;
}

}  // namespace lynx
