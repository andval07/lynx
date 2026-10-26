#include "lynx/vocabulary.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

namespace lynx {

namespace {

constexpr int kDescBytes = 32;
constexpr int kMaxKmeansIters = 10;
constexpr char kMagic[8] = {'L', 'Y', 'N', 'X', 'V', 'O', 'C', '1'};

int hamming(const uint8_t* a, const uint8_t* b) {
  int d = 0;
  for (int w = 0; w < kDescBytes; w += 8) {
    uint64_t x, y;
    std::memcpy(&x, a + w, 8);
    std::memcpy(&y, b + w, 8);
    d += static_cast<int>(std::bitset<64>(x ^ y).count());
  }
  return d;
}

// Bitwise majority of the descriptors: the median in Hamming space.
std::array<uint8_t, kDescBytes> majority(const std::vector<const uint8_t*>& d) {
  std::array<int, 8 * kDescBytes> ones{};
  for (const uint8_t* p : d) {
    for (int b = 0; b < 8 * kDescBytes; ++b) ones[b] += (p[b / 8] >> (b % 8)) & 1;
  }
  std::array<uint8_t, kDescBytes> c{};
  for (int b = 0; b < 8 * kDescBytes; ++b) {
    if (2 * ones[b] > static_cast<int>(d.size())) c[b / 8] |= uint8_t(1u << (b % 8));
  }
  return c;
}

// k-means++ seeding: each new center is drawn with probability ~ d^2 to the
// closest center already chosen.
std::vector<const uint8_t*> seed_centers(const std::vector<const uint8_t*>& d,
                                         int k, std::mt19937& rng) {
  std::vector<const uint8_t*> c{d[std::uniform_int_distribution<size_t>(0, d.size() - 1)(rng)]};
  std::vector<double> dist2(d.size(), std::numeric_limits<double>::max());
  while (static_cast<int>(c.size()) < k) {
    for (size_t i = 0; i < d.size(); ++i) {
      const double h = hamming(d[i], c.back());
      dist2[i] = std::min(dist2[i], h * h);
    }
    // All left descriptors equal a center: fewer clusters than k.
    if (*std::max_element(dist2.begin(), dist2.end()) == 0) break;
    std::discrete_distribution<size_t> pick(dist2.begin(), dist2.end());
    c.push_back(d[pick(rng)]);
  }
  return c;
}

template <typename T>
void write_pod(std::ofstream& out, const T& v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
T read_pod(std::ifstream& in) {
  T v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!in) throw std::runtime_error("vocabulary file is truncated");
  return v;
}

}  // namespace

double bow_score(const BowVector& a, const BowVector& b) {
  // For L1-normalized vectors 1 - |a-b|/2 = sum over shared words of
  // (|a_i| + |b_i| - |a_i - b_i|) / 2, so only shared words matter.
  double s = 0;
  auto ia = a.begin(), ib = b.begin();
  while (ia != a.end() && ib != b.end()) {
    if (ia->first < ib->first) {
      ++ia;
    } else if (ib->first < ia->first) {
      ++ib;
    } else {
      s += std::abs(ia->second) + std::abs(ib->second) - std::abs(ia->second - ib->second);
      ++ia;
      ++ib;
    }
  }
  return 0.5 * s;
}

void Vocabulary::build(int node, const std::vector<const uint8_t*>& descs,
                       int level, int k, int levels, std::mt19937& rng) {
  if (level == levels || descs.empty()) {
    nodes_[node].word = static_cast<int>(idf_.size());
    idf_.push_back(0);
    return;
  }
  // Few descriptors left: each one becomes its own cluster.
  std::vector<std::vector<const uint8_t*>> clusters;
  if (static_cast<int>(descs.size()) <= k) {
    for (const uint8_t* d : descs) clusters.push_back({d});
  } else {
    std::vector<Descriptor> centers;
    for (const uint8_t* c : seed_centers(descs, k, rng)) {
      Descriptor cd;
      std::memcpy(cd.data(), c, kDescBytes);
      centers.push_back(cd);
    }
    const int n_centers = static_cast<int>(centers.size());
    std::vector<int> assign(descs.size(), -1);
    for (int it = 0; it < kMaxKmeansIters; ++it) {
      bool changed = false;
      for (size_t i = 0; i < descs.size(); ++i) {
        int best = 0, best_d = std::numeric_limits<int>::max();
        for (int c = 0; c < n_centers; ++c) {
          const int h = hamming(descs[i], centers[c].data());
          if (h < best_d) best_d = h, best = c;
        }
        if (assign[i] != best) assign[i] = best, changed = true;
      }
      clusters.assign(n_centers, {});
      for (size_t i = 0; i < descs.size(); ++i) clusters[assign[i]].push_back(descs[i]);
      if (!changed) break;
      for (int c = 0; c < n_centers; ++c) {
        if (!clusters[c].empty()) centers[c] = majority(clusters[c]);
      }
    }
    clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
                                  [](const auto& c) { return c.empty(); }),
                   clusters.end());
  }
  for (const auto& cl : clusters) {
    const int child = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    nodes_[child].center = majority(cl);
    nodes_[node].children.push_back(child);
    build(child, cl, level + 1, k, levels, rng);
  }
}

Vocabulary Vocabulary::train(const std::vector<cv::Mat>& images, int k,
                             int levels, std::mt19937& rng) {
  if (k < 2 || levels < 1) throw std::invalid_argument("vocabulary needs k >= 2, levels >= 1");
  std::vector<const uint8_t*> all;
  for (const cv::Mat& m : images) {
    if (m.empty()) continue;
    if (m.type() != CV_8U || m.cols != kDescBytes) {
      throw std::invalid_argument("vocabulary expects 32-byte ORB descriptors");
    }
    for (int r = 0; r < m.rows; ++r) all.push_back(m.ptr<uint8_t>(r));
  }
  if (all.empty()) throw std::invalid_argument("no descriptors to train on");

  Vocabulary v;
  v.nodes_.emplace_back();
  v.build(0, all, 0, k, levels, rng);

  // idf = log(N / n_w), n_w = training images that contain word w.
  std::vector<int> n_w(v.idf_.size(), 0);
  int n_images = 0;
  for (const cv::Mat& m : images) {
    if (m.empty()) continue;
    ++n_images;
    std::vector<bool> seen(v.idf_.size(), false);
    for (int r = 0; r < m.rows; ++r) {
      const int w = v.word_of(m.ptr<uint8_t>(r));
      if (!seen[w]) seen[w] = true, ++n_w[w];
    }
  }
  for (size_t w = 0; w < v.idf_.size(); ++w) {
    v.idf_[w] = n_w[w] > 0 ? std::log(double(n_images) / n_w[w]) : 0.0;
  }
  return v;
}

int Vocabulary::word_of(const uint8_t* d) const {
  int node = 0;
  while (!nodes_[node].children.empty()) {
    int best = nodes_[node].children.front(), best_d = std::numeric_limits<int>::max();
    for (int c : nodes_[node].children) {
      const int h = hamming(d, nodes_[c].center.data());
      if (h < best_d) best_d = h, best = c;
    }
    node = best;
  }
  return nodes_[node].word;
}

BowVector Vocabulary::transform(const cv::Mat& descriptors) const {
  std::map<int, double> tf;
  for (int r = 0; r < descriptors.rows; ++r) {
    const int w = word_of(descriptors.ptr<uint8_t>(r));
    if (idf_[w] > 0) tf[w] += idf_[w];
  }
  double norm = 0;
  for (const auto& [w, x] : tf) norm += x;
  BowVector v(tf.begin(), tf.end());
  if (norm > 0) {
    for (auto& [w, x] : v) x /= norm;
  }
  return v;
}

void Vocabulary::save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  out.write(kMagic, sizeof(kMagic));
  write_pod(out, static_cast<int32_t>(nodes_.size()));
  for (const Node& n : nodes_) {
    out.write(reinterpret_cast<const char*>(n.center.data()), kDescBytes);
    write_pod(out, static_cast<int32_t>(n.word));
    write_pod(out, static_cast<int32_t>(n.children.size()));
    for (int c : n.children) write_pod(out, static_cast<int32_t>(c));
  }
  write_pod(out, static_cast<int32_t>(idf_.size()));
  for (double x : idf_) write_pod(out, x);
}

Vocabulary Vocabulary::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  char magic[sizeof(kMagic)];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error(path + " is not a lynx vocabulary");
  }
  Vocabulary v;
  const int n_nodes = read_pod<int32_t>(in);
  if (n_nodes <= 0) throw std::runtime_error("vocabulary has no nodes");
  v.nodes_.resize(n_nodes);
  for (Node& n : v.nodes_) {
    in.read(reinterpret_cast<char*>(n.center.data()), kDescBytes);
    n.word = read_pod<int32_t>(in);
    const int n_children = read_pod<int32_t>(in);
    for (int c = 0; c < n_children; ++c) {
      const int child = read_pod<int32_t>(in);
      if (child <= 0 || child >= n_nodes) throw std::runtime_error("bad child index");
      n.children.push_back(child);
    }
  }
  const int n_words = read_pod<int32_t>(in);
  v.idf_.resize(n_words);
  for (double& x : v.idf_) x = read_pod<double>(in);
  for (const Node& n : v.nodes_) {
    if (n.children.empty() && (n.word < 0 || n.word >= n_words)) {
      throw std::runtime_error("bad word index");
    }
  }
  return v;
}

}  // namespace lynx
