// Runs RGB-D odometry on a TUM sequence and writes the trajectory in TUM
// format. Prints a quick ATE if the folder has groundtruth.txt; the numbers in
// the README come from evo on the written file.

#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

#include "lynx/odometry.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: lynx_vo <tum_sequence_dir> <out.txt> [max_frames]\n");
    return 2;
  }
  const std::string dir = argv[1];
  const std::string out = argv[2];
  const int max_frames = argc > 3 ? std::stoi(argv[3]) : 0;
  try {
    const lynx::RunStats s = lynx::run_tum_sequence(dir, max_frames);
    lynx::write_tum_trajectory(out, s.trajectory);
    for (size_t i = 0; i < s.per_frame.size(); ++i) {
      const auto& r = s.per_frame[i];
      if (!r.tracked) std::fprintf(stderr, "frame %zu lost (%d matches)\n", i, r.matches);
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
