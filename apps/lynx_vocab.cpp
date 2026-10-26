// Trains the ORB vocabulary used for loop detection from one or more TUM
// sequences. One image every `step` frames is enough: neighbouring frames see
// almost the same words.

#include <cstdio>
#include <exception>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "lynx/frontend.hpp"
#include "lynx/vocabulary.hpp"

namespace {

constexpr int kStep = 10;
constexpr int kBranching = 10;
constexpr int kLevels = 4;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: lynx_vocab <out.bin> <tum_sequence_dir>...\n");
    return 2;
  }
  try {
    std::vector<cv::Mat> descs;
    for (int a = 2; a < argc; ++a) {
      const std::string dir = argv[a];
      const lynx::OrbFrontend frontend(lynx::tum_calibration(dir));
      const auto frames = lynx::load_tum_sequence(dir);
      for (size_t i = 0; i < frames.size(); i += kStep) {
        const cv::Mat gray = cv::imread(dir + "/" + frames[i].rgb, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) throw std::runtime_error("cannot read " + frames[i].rgb);
        descs.push_back(frontend.extract(gray).descriptors);
      }
      std::printf("%s: %zu images\n", dir.c_str(), (frames.size() + kStep - 1) / kStep);
    }
    std::mt19937 rng(7);
    const lynx::Vocabulary voc = lynx::Vocabulary::train(descs, kBranching, kLevels, rng);
    voc.save(argv[1]);
    std::printf("%d words from %zu images -> %s\n", voc.n_words(), descs.size(), argv[1]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
