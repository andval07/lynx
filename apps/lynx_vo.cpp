// Runs RGB-D odometry on a TUM sequence and writes the trajectory in TUM
// format. Prints a quick ATE if the folder has groundtruth.txt; the numbers in
// the README come from evo on the written file.

#include <cstdio>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "lynx/mono_odometry.hpp"
#include "lynx/odometry.hpp"
#include "lynx/tum.hpp"

namespace {

// Monocular run on the RGB images only. Frames before the initialization have
// no pose and are not written; the scale is arbitrary, so evaluate with a
// Sim(3) alignment.
lynx::Trajectory run_mono(const std::string& dir, int max_frames) {
  auto frames = lynx::load_tum_sequence(dir);
  if (max_frames > 0 && static_cast<int>(frames.size()) > max_frames) {
    frames.resize(max_frames);
  }
  lynx::MonoOdometry vo(lynx::tum_calibration(dir));
  int lost = 0;
  for (size_t i = 0; i < frames.size(); ++i) {
    const cv::Mat gray = cv::imread(dir + "/" + frames[i].rgb, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) throw std::runtime_error("cannot read " + frames[i].rgb);
    const lynx::FrameResult r = vo.track(gray);
    if (vo.initialized() && !r.tracked) ++lost;
  }
  lynx::Trajectory traj;
  if (!vo.initialized()) return traj;
  const auto poses = vo.refined_poses();
  for (size_t i = 0; i < poses.size(); ++i) {
    if (vo.has_pose(i)) traj.push_back({frames[i].stamp, poses[i]});
  }
  std::printf("frames %zu, posed %zu, keyframes %d, lost %d\n", frames.size(),
              traj.size(), vo.keyframes(), lost);
  return traj;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  lynx::OdometryParams params;
  bool mono = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--mono") {
      mono = true;
    } else if (a == "--no-ba") {
      params.local_ba = false;
    } else if (a == "--vocab" && i + 1 < argc) {
      params.vocabulary = argv[++i];
    } else {
      pos.push_back(a);
    }
  }
  if (pos.size() < 2) {
    std::fprintf(stderr,
                 "usage: lynx_vo <tum_sequence_dir> <out.txt> [max_frames] [--mono] [--no-ba] [--vocab file]\n");
    return 2;
  }
  const std::string& dir = pos[0];
  const std::string& out = pos[1];
  try {
    const int max_frames = pos.size() > 2 ? std::stoi(pos[2]) : 0;
    if (mono) {
      const lynx::Trajectory traj = run_mono(dir, max_frames);
      if (traj.empty()) throw std::runtime_error("monocular initialization failed");
      lynx::write_tum_trajectory(out, traj);
      return 0;
    }
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
