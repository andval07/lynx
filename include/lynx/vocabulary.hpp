#pragma once

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace lynx {

// Sparse bag of words: (word id, weight) sorted by word id, L1 normalized.
using BowVector = std::vector<std::pair<int, double>>;

// Similarity in [0, 1] of two L1-normalized vectors: 1 - |a - b|_1 / 2.
double bow_score(const BowVector& a, const BowVector& b);

// Vocabulary tree on 256-bit ORB descriptors, built with k-medians at every
// level (bitwise majority as the center) as in DBoW2. Leaves are the words,
// weighted with tf-idf.
class Vocabulary {
 public:
  // images: one descriptor matrix (CV_8U, 32 columns) per training image.
  static Vocabulary train(const std::vector<cv::Mat>& images, int k, int levels,
                          std::mt19937& rng);
  static Vocabulary load(const std::string& path);
  void save(const std::string& path) const;

  BowVector transform(const cv::Mat& descriptors) const;
  int word_of(const uint8_t* descriptor) const;

  int n_words() const { return static_cast<int>(idf_.size()); }
  bool empty() const { return nodes_.empty(); }

 private:
  using Descriptor = std::array<uint8_t, 32>;
  struct Node {
    Descriptor center{};
    std::vector<int> children;
    int word = -1;
  };

  void build(int node, const std::vector<const uint8_t*>& descs, int level,
             int k, int levels, std::mt19937& rng);

  std::vector<Node> nodes_;  // nodes_[0] is the root
  std::vector<double> idf_;  // per word
};

}  // namespace lynx
