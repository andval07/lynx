#pragma once

#include <array>
#include <string>
#include <vector>

#include "lynx/camera.hpp"
#include "lynx/trajectory.hpp"

namespace lynx {

// One RGB-D pair of a TUM sequence, paths relative to the sequence folder.
struct TumFrame {
  double stamp = 0;
  std::string rgb;
  std::string depth;
};

struct TumCalibration {
  Camera cam;
  std::array<double, 5> dist{};  // k1 k2 p1 p2 k3, OpenCV order
  double depth_scale = 5000.0;   // png value per meter
};

// Reads rgb.txt and depth.txt and pairs each rgb image with the nearest depth
// image within max_dt seconds. Throws std::runtime_error if files are missing.
std::vector<TumFrame> load_tum_sequence(const std::string& dir,
                                        double max_dt = 0.02);

// Calibration published for the freiburg1/2/3 cameras, picked from the
// sequence name. Throws std::runtime_error for unknown names.
TumCalibration tum_calibration(const std::string& dir);

// "timestamp tx ty tz qx qy qz qw" per line, '#' lines ignored.
Trajectory read_tum_trajectory(const std::string& path);
void write_tum_trajectory(const std::string& path, const Trajectory& traj);

}  // namespace lynx
