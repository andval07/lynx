#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>

#include "lynx/vocabulary.hpp"

using namespace lynx;

namespace {

// Descriptors drawn around fixed random centers, with a few flipped bits, so
// the true clusters are known.
struct Synthetic {
  std::vector<cv::Mat> centers;  // one 1x32 row each
};

Synthetic make_centers(std::mt19937& rng, int n) {
  Synthetic s;
  std::uniform_int_distribution<int> byte(0, 255);
  for (int c = 0; c < n; ++c) {
    cv::Mat m(1, 32, CV_8U);
    for (int b = 0; b < 32; ++b) m.at<uint8_t>(0, b) = static_cast<uint8_t>(byte(rng));
    s.centers.push_back(m);
  }
  return s;
}

// An image that sees the given clusters, `per_cluster` noisy descriptors each.
cv::Mat make_image(std::mt19937& rng, const Synthetic& s,
                   const std::vector<int>& clusters, int per_cluster) {
  std::uniform_int_distribution<int> bit(0, 255);
  cv::Mat img(0, 32, CV_8U);
  for (int c : clusters) {
    for (int k = 0; k < per_cluster; ++k) {
      cv::Mat d = s.centers[c].clone();
      for (int f = 0; f < 6; ++f) {
        const int b = bit(rng);
        d.at<uint8_t>(0, b / 8) ^= static_cast<uint8_t>(1u << (b % 8));
      }
      img.push_back(d);
    }
  }
  return img;
}

std::vector<int> range(int a, int b) {
  std::vector<int> v;
  for (int i = a; i < b; ++i) v.push_back(i);
  return v;
}

}  // namespace

TEST(Vocabulary, SameClustersScoreHigherThanOthers) {
  std::mt19937 rng(1);
  const Synthetic s = make_centers(rng, 60);
  std::vector<cv::Mat> train;
  for (int i = 0; i < 6; ++i) train.push_back(make_image(rng, s, range(10 * i, 10 * i + 10), 20));
  const Vocabulary voc = Vocabulary::train(train, 8, 3, rng);
  EXPECT_GT(voc.n_words(), 30);

  const BowVector a = voc.transform(make_image(rng, s, range(0, 10), 15));
  const BowVector a2 = voc.transform(make_image(rng, s, range(0, 10), 15));
  const BowVector b = voc.transform(make_image(rng, s, range(30, 40), 15));
  EXPECT_NEAR(bow_score(a, a), 1.0, 1e-12);
  EXPECT_GT(bow_score(a, a2), 0.5);
  EXPECT_LT(bow_score(a, b), 0.1);
}

TEST(Vocabulary, VectorsAreL1Normalized) {
  std::mt19937 rng(2);
  const Synthetic s = make_centers(rng, 20);
  std::vector<cv::Mat> train;
  for (int i = 0; i < 4; ++i) train.push_back(make_image(rng, s, range(5 * i, 5 * i + 5), 10));
  const Vocabulary voc = Vocabulary::train(train, 5, 2, rng);
  double sum = 0;
  for (const auto& [w, x] : voc.transform(train[0])) sum += x;
  EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(Vocabulary, SaveLoadRoundTrip) {
  std::mt19937 rng(3);
  const Synthetic s = make_centers(rng, 30);
  std::vector<cv::Mat> train;
  for (int i = 0; i < 3; ++i) train.push_back(make_image(rng, s, range(10 * i, 10 * i + 10), 10));
  const Vocabulary voc = Vocabulary::train(train, 6, 3, rng);
  const std::string path =
      (std::filesystem::temp_directory_path() / "lynx_test_vocab.bin").string();
  voc.save(path);
  const Vocabulary back = Vocabulary::load(path);
  std::filesystem::remove(path);

  EXPECT_EQ(back.n_words(), voc.n_words());
  const cv::Mat q = make_image(rng, s, range(0, 30), 3);
  EXPECT_EQ(back.transform(q), voc.transform(q));
}

TEST(Vocabulary, RejectsBadFile) {
  const std::string path =
      (std::filesystem::temp_directory_path() / "lynx_test_not_vocab.bin").string();
  { std::ofstream(path) << "hello"; }
  EXPECT_THROW(Vocabulary::load(path), std::runtime_error);
  std::filesystem::remove(path);
}
