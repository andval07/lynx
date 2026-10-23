#include "lynx/se3.hpp"

#include <algorithm>
#include <cmath>

namespace lynx {

namespace {
constexpr double kSmallAngle = 1e-6;
constexpr double kNearPi = 1e-4;
}  // namespace

Mat3 hat(const Vec3& w) {
  Mat3 m;
  m << 0, -w.z(), w.y(),
       w.z(), 0, -w.x(),
       -w.y(), w.x(), 0;
  return m;
}

// Rodrigues: R = I + sin(t)/t W + (1 - cos(t))/t^2 W^2
Mat3 so3_exp(const Vec3& w) {
  const double theta = w.norm();
  const Mat3 W = hat(w);
  if (theta < kSmallAngle) {
    return Mat3::Identity() + W + 0.5 * W * W;
  }
  return Mat3::Identity() + std::sin(theta) / theta * W +
         (1.0 - std::cos(theta)) / (theta * theta) * W * W;
}

Vec3 so3_log(const Mat3& R) {
  const Vec3 v(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  // atan2 keeps full precision near 0 and pi, where acos does not.
  const double c = std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0);
  const double theta = std::atan2(0.5 * v.norm(), c);
  if (theta < kSmallAngle) {
    return 0.5 * v;
  }
  if (M_PI - theta < kNearPi) {
    // sin(theta) is almost zero, so v carries no axis. The symmetric part
    // of R + I is (1 + cos) I + (1 - cos) a a^T ~ 2 a a^T.
    const Mat3 B = 0.25 * (R + R.transpose()) + 0.5 * Mat3::Identity();
    int k = 0;
    B.diagonal().maxCoeff(&k);
    Vec3 axis = B.col(k) / std::sqrt(B(k, k));
    // Pick the sign that agrees with the small antisymmetric part.
    if (axis.dot(v) < 0) axis = -axis;
    return theta * axis.normalized();
  }
  return theta / (2.0 * std::sin(theta)) * v;
}

Mat3 so3_left_jacobian(const Vec3& w) {
  const double theta = w.norm();
  const Mat3 W = hat(w);
  if (theta < kSmallAngle) {
    return Mat3::Identity() + 0.5 * W + W * W / 6.0;
  }
  const double t2 = theta * theta;
  return Mat3::Identity() + (1.0 - std::cos(theta)) / t2 * W +
         (theta - std::sin(theta)) / (t2 * theta) * W * W;
}

Mat3 so3_left_jacobian_inv(const Vec3& w) {
  const double theta = w.norm();
  const Mat3 W = hat(w);
  if (theta < kSmallAngle) {
    return Mat3::Identity() - 0.5 * W + W * W / 12.0;
  }
  const double t2 = theta * theta;
  const double k =
      1.0 / t2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
  return Mat3::Identity() - 0.5 * W + k * W * W;
}

SE3::SE3(const Mat3& R, const Vec3& t)
    : R_(Eigen::Quaterniond(R).normalized().toRotationMatrix()), t_(t) {}

SE3::SE3(const Eigen::Quaterniond& q, const Vec3& t)
    : R_(q.normalized().toRotationMatrix()), t_(t) {}

SE3 SE3::exp(const Vec6& xi) {
  const Vec3 rho = xi.head<3>();
  const Vec3 phi = xi.tail<3>();
  return SE3(so3_exp(phi), so3_left_jacobian(phi) * rho);
}

Vec6 SE3::log() const {
  const Vec3 phi = so3_log(R_);
  Vec6 xi;
  xi << so3_left_jacobian_inv(phi) * t_, phi;
  return xi;
}

SE3 SE3::inverse() const {
  const Mat3 Rt = R_.transpose();
  return SE3(Rt, -Rt * t_);
}

SE3 SE3::operator*(const SE3& other) const {
  return SE3(R_ * other.R_, R_ * other.t_ + t_);
}

Vec3 SE3::operator*(const Vec3& p) const { return R_ * p + t_; }

Mat6 SE3::adjoint() const {
  Mat6 A = Mat6::Zero();
  A.block<3, 3>(0, 0) = R_;
  A.block<3, 3>(0, 3) = hat(t_) * R_;
  A.block<3, 3>(3, 3) = R_;
  return A;
}

Eigen::Matrix4d SE3::matrix() const {
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m.block<3, 3>(0, 0) = R_;
  m.block<3, 1>(0, 3) = t_;
  return m;
}

}  // namespace lynx
