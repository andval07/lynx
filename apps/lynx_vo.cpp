// Runs RGB-D odometry on a TUM sequence and writes the trajectory in TUM
// format. Prints a quick ATE if the folder has groundtruth.txt; the numbers in
// the README come from evo on the written file.

#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "lynx/odometry.hpp"

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  lynx::OdometryParams params;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--no-ba") {
      params.local_ba = false;
    } else if (a == "--vocab" && i + 1 < argc) {
      params.vocabulary = argv[++i];
    } else {
      pos.push_back(a);
    }
  }
  if (pos.size() < 2) {
    std::fprintf(stderr,
                 "usage: lynx_vo <tum_sequence_dir> <out.txt> [max_frames] [--no-ba] [--vocab file]\n");
    return 2;
  }
  const std::string& dir = pos[0];
  const std::string& out = pos[1];
  try {
    const int max_frames = pos.size() > 2 ? std::stoi(pos[2]) : 0;
    const lynx::RunStats s = lynx::run_tum_sequence(dir, max_frames, params);
    lynx::write_tum_trajectory(out, s.trajectory);
    for (size_t i = 0; i < s.per_frame.size(); ++i) {
      const auto& r = s.per_frame[i];
      if (!r.tracked) std::fprintf(stderr, "frame %zu lost (%d matches)\n", i, r.matches);
    }
    for (const auto& l : s.loops) {
      std::printf("loop kf %d -> kf %d, %d inliers, score %.3f\n", l.cur, l.cand,
                  l.inliers, l.score);
    }
    std::printf("frames %d, keyframes %d, lost %d, %.1f ms/frame\n", s.frames,
                s.keyframes, s.lost, s.ms_per_frame);
    const std::string gt = dir + "/groundtruth.txt";
    if (std::filesystem::exists(gt)) {
      const double ate = lynx::ate_rmse(s.trajectory, lynx::read_tum_trajectory(gt));
      std::printf("ATE RMSE %.4f m\n", ate);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
