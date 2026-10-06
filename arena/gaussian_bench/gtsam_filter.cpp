// GTSAM (C++) filters on a problem written by run.py, in the same output format as driver.c.
//   kf: gtsam::KalmanFilter on the whole state as one vector (predictQ, updateQ)
//   fg: a factor graph with one key per block and time step; each step adds the transition and
//       observation factors, eliminates the previous step's keys (eliminatePartialSequential),
//       and reads every block's mean and covariance from the remaining graph's Bayes tree.
// Full covariances are whitened once (the noise models get unit covariance); per step only the
// right-hand side of an observation is whitened.
extern "C" {
#include "io.h"
}
#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/KalmanFilter.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

using namespace gtsam;

static Matrix mat(const double* p, int r, int c) {
  Matrix M(r, c);
  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++) M(i, j) = p[i * c + j];
  return M;
}
static Vector vec(const double* p, int n) { return Eigen::Map<const Vector>(p, n); }

struct Filter {
  virtual void head(const double* const* ys, double* out) = 0;
  virtual void step(const double* const* ys, double* out) = 0;
  virtual ~Filter() {}
};

struct KF : Filter {
  const Problem& P;
  int N = 0, M = 0;
  std::vector<int> off;
  Vector mu0, b, c;
  Matrix P0, A, Q, H, R;
  KalmanFilter kf;
  KalmanFilter::State st;
  static int total(const Problem& P) {
    int n = 0;
    for (int i = 0; i < P.nb; i++) n += P.dim[i];
    return n;
  }
  explicit KF(const Problem& P_) : P(P_), kf(total(P_)) {
    off.push_back(0);
    for (int i = 0; i < P.nb; i++) off.push_back(off.back() + P.dim[i]);
    N = off.back();
    for (int i = 0; i < P.nf; i++)
      if (P.f[i].kind == 2) M += P.f[i].rows;
    mu0 = Vector::Zero(N); b = Vector::Zero(N); c = Vector::Zero(M);
    P0 = Matrix::Zero(N, N); A = Matrix::Zero(N, N); Q = Matrix::Zero(N, N);
    H = Matrix::Zero(M, N); R = Matrix::Zero(M, M);
    int r = 0;
    for (int i = 0; i < P.nf; i++) {
      const Fac& f = P.f[i];
      int d = f.rows;
      if (f.kind == 0) {
        mu0.segment(off[f.child], d) = vec(f.b, d);
        P0.block(off[f.child], off[f.child], d, d) = mat(f.Q, d, d);
      } else if (f.kind == 1) {
        for (int k = 0; k < f.np; k++)
          A.block(off[f.child], off[f.parent[k]], d, P.dim[f.parent[k]]) += mat(f.A[k], d, P.dim[f.parent[k]]);
        b.segment(off[f.child], d) = vec(f.b, d);
        Q.block(off[f.child], off[f.child], d, d) = mat(f.Q, d, d);
      } else {
        for (int k = 0; k < f.np; k++)
          H.block(r, off[f.parent[k]], d, P.dim[f.parent[k]]) += mat(f.A[k], d, P.dim[f.parent[k]]);
        c.segment(r, d) = vec(f.b, d);
        R.block(r, r, d, d) = mat(f.Q, d, d);
        r += d;
      }
    }
  }
  Vector z(const double* const* ys) {
    Vector y(M);
    int r = 0, k = 0;
    for (int i = 0; i < P.nf; i++)
      if (P.f[i].kind == 2) {
        y.segment(r, P.f[i].rows) = vec(ys[k++], P.f[i].rows);
        r += P.f[i].rows;
      }
    return y - c;
  }
  void write(double* out) {
    Vector m = st->mean();
    Matrix S = st->covariance();
    int o = 0;
    for (int i = 0; i < P.nb; i++)
      for (int j = 0; j < P.dim[i]; j++) out[o++] = m(off[i] + j);
    for (int i = 0; i < P.nb; i++)
      for (int r = 0; r < P.dim[i]; r++)
        for (int j = 0; j < P.dim[i]; j++) out[o++] = S(off[i] + r, off[i] + j);
  }
  void head(const double* const* ys, double* out) override {
    st = kf.updateQ(kf.init(mu0, P0), H, z(ys), R);
    write(out);
  }
  void step(const double* const* ys, double* out) override {
    st = kf.updateQ(kf.predictQ(st, A, Matrix::Identity(N, N), b, Q), H, z(ys), R);
    write(out);
  }
};

struct FG : Filter {
  const Problem& P;
  struct W {  // a whitened factor: rows of L^-1 [terms | rhs]
    Matrix Li;
    std::vector<Matrix> A;  // whitened, per parent
    Matrix I;               // whitened identity on the child (prior and cond)
    Vector rhs;             // whitened b (prior and cond)
  };
  std::vector<W> w;
  GaussianFactorGraph::shared_ptr msg;
  int t = 0;
  explicit FG(const Problem& P_) : P(P_) {
    for (int i = 0; i < P.nf; i++) {
      const Fac& f = P.f[i];
      W x;
      Matrix L = mat(f.Q, f.rows, f.rows).llt().matrixL();
      x.Li = L.inverse();
      double sign = f.kind == 2 ? 1 : -1;  // cond: child - A p = b
      for (int k = 0; k < f.np; k++) x.A.push_back(sign * x.Li * mat(f.A[k], f.rows, P.dim[f.parent[k]]));
      x.I = x.Li;
      x.rhs = x.Li * vec(f.b, f.rows);
      w.push_back(x);
    }
  }
  Key key(int block, int time) const { return Key((time & 1) * P.nb + block); }
  void add(GaussianFactorGraph& g, const double* const* ys, bool first) {
    int k = 0;
    for (int i = 0; i < P.nf; i++) {
      const Fac& f = P.f[i];
      if (f.kind == 2) {
        std::vector<std::pair<Key, Matrix>> terms;
        for (int j = 0; j < f.np; j++) terms.emplace_back(key(f.parent[j], t), w[i].A[j]);
        g.emplace_shared<JacobianFactor>(terms, w[i].Li * (vec(ys[k++], f.rows) - vec(f.b, f.rows)));
      } else if ((f.kind == 0) == first) {
        std::vector<std::pair<Key, Matrix>> terms{{key(f.child, t), w[i].I}};
        for (int j = 0; j < f.np; j++) terms.emplace_back(key(f.parent[j], t + f.ptime[j]), w[i].A[j]);
        g.emplace_shared<JacobianFactor>(terms, w[i].rhs);
      }
    }
  }
  void write(double* out) {
    auto bt = msg->eliminateMultifrontal();
    VectorValues m = bt->optimize();
    int o = 0;
    for (int i = 0; i < P.nb; i++) {
      const Vector& v = m.at(key(i, t));
      for (int j = 0; j < P.dim[i]; j++) out[o++] = v(j);
    }
    for (int i = 0; i < P.nb; i++) {
      Matrix S = bt->marginalCovariance(key(i, t));
      for (int r = 0; r < P.dim[i]; r++)
        for (int j = 0; j < P.dim[i]; j++) out[o++] = S(r, j);
    }
  }
  void head(const double* const* ys, double* out) override {
    t = 0;
    auto g = std::make_shared<GaussianFactorGraph>();
    add(*g, ys, true);
    msg = g;
    write(out);
  }
  void step(const double* const* ys, double* out) override {
    t++;
    GaussianFactorGraph g = *msg;
    add(g, ys, false);
    Ordering prev;
    for (int i = 0; i < P.nb; i++) prev.push_back(key(i, t - 1));
    msg = g.eliminatePartialSequential(prev).second;
    write(out);
  }
};

int main(int argc, char** argv) {
  Problem P = read_problem(argv[1]);
  std::unique_ptr<Filter> F;
  if (std::string(argv[2]) == "kf") F.reset(new KF(P));
  else F.reset(new FG(P));
  std::vector<double> out(P.out_size);
  auto run_all = [&](long reps) {
    for (long r = 0; r < reps; r++) {
      F->head(P.ys, out.data());
      for (int t = 1; t < P.T; t++) F->step(P.ys + t * P.nobs, out.data());
    }
  };
  for (int t = 0; t < P.Tc; t++) {
    if (t == 0) F->head(P.ys, out.data());
    else F->step(P.ys + t * P.nobs, out.data());
    for (double x : out) std::printf("%.17g ", x);
  }
  std::printf("\n");
  using clk = std::chrono::steady_clock;
  long reps = 1;
  for (;;) {
    auto t0 = clk::now();
    run_all(reps);
    if (std::chrono::duration<double>(clk::now() - t0).count() >= 0.01) break;
    reps *= 2;
  }
  std::vector<double> s;
  for (int k = 0; k < 7; k++) {
    auto t0 = clk::now();
    run_all(reps);
    s.push_back(std::chrono::duration<double>(clk::now() - t0).count() / (reps * double(P.T)));
  }
  std::sort(s.begin(), s.end());
  std::printf("%.6e\n", s[3]);
}
