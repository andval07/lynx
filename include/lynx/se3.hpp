#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lynx {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat6 = Eigen::Matrix<double, 6, 6>;

// Skew-symmetric matrix: hat(w) * v == w.cross(v).
Mat3 hat(const Vec3& w);

Mat3 so3_exp(const Vec3& w);
Vec3 so3_log(const Mat3& R);
Mat3 so3_left_jacobian(const Vec3& w);
Mat3 so3_left_jacobian_inv(const Vec3& w);

// Rigid transform p' = R p + t.
// Tangent vectors are xi = (rho, phi): translation part first, rotation second.
class SE3 {
 public:
  SE3() : R_(Mat3::Identity()), t_(Vec3::Zero()) {}
  SE3(const Mat3& R, const Vec3& t);
  SE3(const Eigen::Quaterniond& q, const Vec3& t);

  static SE3 exp(const Vec6& xi);
  Vec6 log() const;

  SE3 inverse() const;
  SE3 operator*(const SE3& other) const;
  Vec3 operator*(const Vec3& p) const;

  // Ad(T) xi == log(T exp(xi) T^-1)
  Mat6 adjoint() const;

  const Mat3& R() const { return R_; }
  const Vec3& t() const { return t_; }
  Eigen::Quaterniond quaternion() const { return Eigen::Quaterniond(R_); }
  Eigen::Matrix4d matrix() const;

 private:
  Mat3 R_;
  Vec3 t_;
};

}  // namespace lynx
