#include <Eigen/SVD>

#include "lynx/geometry.hpp"

namespace lynx {

Vec3 triangulate(const Mat34& P1, const Mat34& P2, const Vec2& x1,
                 const Vec2& x2) {
  // Each view gives two rows of A X = 0: x * P.row(2) - P.row(0), ...
  Eigen::Matrix4d A;
  A.row(0) = x1.x() * P1.row(2) - P1.row(0);
  A.row(1) = x1.y() * P1.row(2) - P1.row(1);
  A.row(2) = x2.x() * P2.row(2) - P2.row(0);
  A.row(3) = x2.y() * P2.row(2) - P2.row(1);
  // Normalize rows so pixel and normalized inputs are weighted alike.
  for (int i = 0; i < 4; ++i) A.row(i).normalize();
  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
  const Eigen::Vector4d X = svd.matrixV().col(3);
  return X.head<3>() / X(3);
}

Mat34 projection_matrix(const Camera& cam, const SE3& T_cw) {
  Mat34 Rt;
  Rt.block<3, 3>(0, 0) = T_cw.R();
  Rt.col(3) = T_cw.t();
  return cam.K() * Rt;
}

}  // namespace lynx
