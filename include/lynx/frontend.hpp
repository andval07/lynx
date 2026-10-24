#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

#include "lynx/tum.hpp"

namespace lynx {

struct Features {
  std::vector<cv::KeyPoint> keypoints;  // raw image coordinates
  cv::Mat descriptors;                  // one 32-byte ORB row per keypoint
  std::vector<Vec2> uv;                 // undistorted pixel coordinates
};

// ORB detection and description from OpenCV, plus lens undistortion of the
// keypoints so the geometry only sees an ideal pinhole camera.
class OrbFrontend {
 public:
  explicit OrbFrontend(const TumCalibration& calib, int n_features = 1500);

  Features extract(const cv::Mat& gray) const;

  // Brute-force Hamming kNN with Lowe's ratio test, one match per train
  // descriptor (the closest query wins).
  static std::vector<cv::DMatch> match(const cv::Mat& query,
                                       const cv::Mat& train, float ratio);

 private:
  cv::Ptr<cv::ORB> orb_;
  cv::Mat K_;
  cv::Mat dist_;
};

}  // namespace lynx
