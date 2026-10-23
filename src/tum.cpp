#include "lynx/tum.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace lynx {

namespace {

std::vector<std::pair<double, std::string>> read_list(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<std::pair<double, std::string>> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    double t = 0;
    std::string file;
    if (ss >> t >> file) out.emplace_back(t, file);
  }
  return out;
}

}  // namespace

std::vector<TumFrame> load_tum_sequence(const std::string& dir, double max_dt) {
  const auto rgb = read_list(dir + "/rgb.txt");
  const auto depth = read_list(dir + "/depth.txt");
  std::vector<TumFrame> frames;
  size_t j = 0;
  for (const auto& [t, file] : rgb) {
    while (j + 1 < depth.size() &&
           std::abs(depth[j + 1].first - t) <= std::abs(depth[j].first - t)) {
      ++j;
    }
    if (j < depth.size() && std::abs(depth[j].first - t) <= max_dt) {
      frames.push_back({t, file, depth[j].second});
    }
  }
  return frames;
}

TumCalibration tum_calibration(const std::string& dir) {
  TumCalibration c;
  if (dir.find("freiburg1") != std::string::npos) {
    c.cam = {517.3, 516.5, 318.6, 255.3};
    c.dist = {0.2624, -0.9531, -0.0054, 0.0026, 1.1633};
  } else if (dir.find("freiburg2") != std::string::npos) {
    c.cam = {520.9, 521.0, 325.1, 249.7};
    c.dist = {0.2312, -0.7849, -0.0033, -0.0001, 0.9172};
  } else if (dir.find("freiburg3") != std::string::npos) {
    c.cam = {535.4, 539.2, 320.1, 247.6};
  } else {
    throw std::runtime_error("unknown TUM camera for " + dir);
  }
  return c;
}

Trajectory read_tum_trajectory(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  Trajectory traj;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    double t, x, y, z, qx, qy, qz, qw;
    if (ss >> t >> x >> y >> z >> qx >> qy >> qz >> qw) {
      traj.push_back({t, SE3(Eigen::Quaterniond(qw, qx, qy, qz), Vec3(x, y, z))});
    }
  }
  std::sort(traj.begin(), traj.end(),
            [](const auto& a, const auto& b) { return a.stamp < b.stamp; });
  return traj;
}

void write_tum_trajectory(const std::string& path, const Trajectory& traj) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << std::fixed;
  for (const auto& p : traj) {
    const Eigen::Quaterniond q = p.T_wc.quaternion();
    const Vec3& t = p.T_wc.t();
    out << std::setprecision(6) << p.stamp << ' ' << std::setprecision(9)
        << t.x() << ' ' << t.y() << ' ' << t.z() << ' ' << q.x() << ' ' << q.y()
        << ' ' << q.z() << ' ' << q.w() << '\n';
  }
}

}  // namespace lynx
