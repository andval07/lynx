#pragma once

#include "lynx/se3.hpp"

namespace lynx {

// Pinhole camera on undistorted pixel coordinates.
struct Camera {
  double fx = 0, fy = 0, cx = 0, cy = 0;

  Vec2 project(const Vec3& p) const {
    return {fx * p.x() / p.z() + cx, fy * p.y() / p.z() + cy};
  }
  Vec3 backproject(const Vec2& uv, double depth) const {
    return {(uv.x() - cx) / fx * depth, (uv.y() - cy) / fy * depth, depth};
  }
  Vec2 normalize(const Vec2& uv) const {
    return {(uv.x() - cx) / fx, (uv.y() - cy) / fy};
  }
  Eigen::Matrix3d K() const {
    Eigen::Matrix3d k;
    k << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    return k;
  }
};

}  // namespace lynx
