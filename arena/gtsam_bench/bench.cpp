// GTSAM (C++) on one discrete problem: all marginals of a prebuilt DiscreteFactorGraph.
// Input (text, from run.py): n, cards[n], m, then per factor: arity, keys..., table (row-major in
// scope order); then the reference marginals in variable order. Output: one JSON line.
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace gtsam;

template <class F>
static double median_time(F f) {
  using clk = std::chrono::steady_clock;
  f();
  long n = 1;
  for (;;) {
    auto t0 = clk::now();
    for (long i = 0; i < n; ++i) f();
    if (std::chrono::duration<double>(clk::now() - t0).count() >= 0.005) break;
    n *= 4;
  }
  std::vector<double> s;
  for (int r = 0; r < 7; ++r) {
    auto t0 = clk::now();
    for (long i = 0; i < n; ++i) f();
    s.push_back(std::chrono::duration<double>(clk::now() - t0).count() / n);
  }
  std::sort(s.begin(), s.end());
  return s[3];
}

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  size_t n, m;
  in >> n;
  std::vector<DiscreteKey> keys(n);
  for (size_t i = 0; i < n; ++i) {
    size_t c;
    in >> c;
    keys[i] = DiscreteKey(i, c);
  }
  DiscreteFactorGraph g;
  in >> m;
  for (size_t f = 0; f < m; ++f) {
    size_t a, size = 1;
    in >> a;
    DiscreteKeys dk;
    for (size_t j = 0; j < a; ++j) {
      size_t k;
      in >> k;
      dk.push_back(keys[k]);
      size *= keys[k].second;
    }
    std::vector<double> t(size);
    for (auto& x : t) in >> x;
    g.add(dk, t);
  }
  double err = 0, sink = 0;
  {
    DiscreteMarginals mg(g);
    for (auto& k : keys) {
      Vector p = mg.marginalProbabilities(k);
      for (int s = 0; s < p.size(); ++s) {
        double r;
        in >> r;
        err = std::max(err, std::fabs(p(s) - r));
      }
    }
  }
  double t_all = median_time([&] {
    DiscreteMarginals mg(g);
    for (auto& k : keys) sink += mg.marginalProbabilities(k)(0);
  });
  double t_elim = median_time([&] { sink += g.eliminateMultifrontal()->size(); });
  std::printf("{\"max_abs_err\": %.3e, \"time_s\": %.6e, \"eliminate_s\": %.6e, \"sink\": %g}\n", err, t_all,
              t_elim, sink > 0 ? 1.0 : 0.0);
}
