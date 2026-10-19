// Granger causality F tests for every ordered pair of series.
//
// For each ordered pair (cause i, effect j), with T = rows - lag usable
// times: the restricted model regresses effect j on an intercept and its
// own lag past values, the unrestricted one adds cause i's lag past
// values, and F = ((RSS_r - RSS_u) / lag) / (RSS_u / (T - 2 lag - 1)).
//
// By the Frisch-Waugh theorem RSS_r - RSS_u = g' G^{-1} g where, after
// projecting the effect's restricted design out of everything through an
// orthonormal basis Q of it, M = L_i - Q Q' L_i holds the cause's lags with
// the effect's own past removed, G = M'M and g = M'e for the effect's
// residual e. Each effect's Q is built once (Cholesky-QR2 on the GPU,
// Householder on the host for designs too ill-conditioned for it), then
// every cause's test against it comes from a few matrix products: the fast
// form G = L'L - C'C, g = L'e - C'(Q'e) with C = Q'L, which the kernel
// checks pair by pair and falls back from to M formed explicitly where the
// cancellation cannot be trusted (see FAST_SCORE_LIMIT). Everything is
// computed in double; each series is centred and scaled first, which
// changes no F.
//
// The method, its limits and the measurements behind them are those of
// the Python package widestats (granger.py), which this follows.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "Rmath.h"
#include "cuda_runtime.h"

#include "cudaUtils.h"
#include "gpublas.h"
#include "cuseful.h"
#include "granger.h"

namespace {

const double NaN = std::numeric_limits<double>::quiet_NaN();
const double Inf = std::numeric_limits<double>::infinity();

// Condition numbers (Frobenius estimates ||A|| ||A^{-1}||, at least the
// 2-norm ones and at most the number of columns times them) above which a
// test is NaN: of the effect's restricted design, and of a cause's G; and
// the fraction of a cause's lags (squared length) left beside the effect's
// own past below which nothing is left to test. Measured against least
// squares over 1,152 pairs (white noise, AR(0.9999), random walks with and
// without drift, trends, I(2) with and without drift, I(3), mixed; n
// 300-30,000; lag 1-8): realistic series stay below 5e4 and 3e5 with
// errors under 1e-8; within these limits every pair was within 1e-4;
// beyond them errors reach 100%.
const double EFFECT_LIMIT = 1e10, CAUSE_LIMIT = 1e12, LEFT_LIMIT = 1e-20;
// Cholesky-QR2 for designs better conditioned than this (as accurate as
// Householder there, measured); Householder QR for the rest.
const double CHOLESKY_QR_LIMIT = 1e7;
// The fast form's error relative to the explicit one grows with
// cond(G) / left and with RSS_r / RSS_u; over 1,501 resolved pairs it was
// at most about 20 eps times that product, so pairs with the product at or
// above this are recomputed from M (error below 2.2e-10 measured).
const double FAST_SCORE_LIMIT = 1e6;
const int MAXLAG = 16;                 // GRANGER_MAXLAG in granger.cu

template <typename T>
T * upload(const std::vector<T> & h)
{
  T * d = gpuAlloc<T>(h.size());
  CUDART_CALL(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
  return d;
}

template <typename T>
void download(const T * d, std::vector<T> & h)
{
  CUDART_CALL(cudaMemcpy(h.data(), d, h.size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
}

dim3 gridFor(size_t n, int threads)
{
  return dim3((unsigned) ((n + threads - 1) / threads));
}

// Centres and scales each column of a (rows x cols) to unit length; a
// constant column becomes NaN.
void scale(int rows, int cols, const double * a, std::vector<double> & out)
{
  out.assign((size_t) rows * cols, 0.0);
  for (int j = 0; j < cols; j++) {
    const double * col = a + (size_t) j * rows;
    double mean = 0;
    for (int t = 0; t < rows; t++) mean += col[t];
    mean /= rows;
    double ss = 0;
    for (int t = 0; t < rows; t++) ss += (col[t] - mean) * (col[t] - mean);
    double s = std::sqrt(ss);
    for (int t = 0; t < rows; t++)
      out[(size_t) j * rows + t] = (col[t] - mean) / s;
  }
}

// The lag matrix of the scaled series s (rows x cols): T x cols*lag,
// column j*lag + k the value k + 1 steps back of series j.
void lagged(int rows, int cols, int lag, const std::vector<double> & s,
            std::vector<double> & out)
{
  int T = rows - lag;
  out.assign((size_t) T * cols * lag, 0.0);
  for (int j = 0; j < cols; j++)
    for (int k = 0; k < lag; k++) {
      const double * src = s.data() + (size_t) j * rows + lag - k - 1;
      double * dst = out.data() + ((size_t) j * lag + k) * T;
      for (int t = 0; t < T; t++) dst[t] = src[t];
    }
}

// Householder QR of the T x m column-major matrix d (on the host), for the
// designs Cholesky-QR2 cannot handle: Q (T x m) into q, and the Frobenius
// condition estimate of R returned (infinite if R is singular or not
// finite).
double householder(int T, int m, std::vector<double> & d,
                   std::vector<double> & q)
{
  std::vector<double> v((size_t) T * m), tau(m);
  for (int j = 0; j < m; j++) {
    double * col = d.data() + (size_t) j * T;
    double norm = 0;
    for (int t = j; t < T; t++) norm += col[t] * col[t];
    norm = std::sqrt(norm);
    double alpha = col[j] >= 0 ? -norm : norm;
    double * vj = v.data() + (size_t) j * T;
    for (int t = 0; t < T; t++) vj[t] = t < j ? 0 : col[t];
    vj[j] -= alpha;
    double vv = 0;
    for (int t = j; t < T; t++) vv += vj[t] * vj[t];
    tau[j] = vv > 0 ? 2 / vv : 0;
    // apply H_j = I - tau v v' to the remaining columns
    for (int c = j; c < m; c++) {
      double * cc = d.data() + (size_t) c * T, s = 0;
      for (int t = j; t < T; t++) s += vj[t] * cc[t];
      s *= tau[j];
      for (int t = j; t < T; t++) cc[t] -= s * vj[t];
    }
  }
  // R is now the upper triangle of d; Q = H_0 ... H_{m-1} [I; 0]
  q.assign((size_t) T * m, 0.0);
  for (int c = 0; c < m; c++) q[(size_t) c * T + c] = 1;
  for (int j = m - 1; j >= 0; j--) {
    const double * vj = v.data() + (size_t) j * T;
    for (int c = 0; c < m; c++) {
      double * qc = q.data() + (size_t) c * T, s = 0;
      for (int t = j; t < T; t++) s += vj[t] * qc[t];
      s *= tau[j];
      for (int t = j; t < T; t++) qc[t] -= s * vj[t];
    }
  }
  // ||R||_F ||R^{-1}||_F by back substitution against the identity
  double rf = 0, rif = 0;
  for (int j = 0; j < m; j++)
    for (int i = 0; i <= j; i++) rf += d[i + (size_t) j * T] * d[i + (size_t) j * T];
  for (int c = 0; c < m; c++) {
    std::vector<double> x(m, 0.0);
    for (int i = m - 1; i >= 0; i--) {
      double s = i == c ? 1 : 0;
      for (int k = i + 1; k < m; k++) s -= d[i + (size_t) k * T] * x[k];
      x[i] = s / d[i + (size_t) i * T];
      rif += x[i] * x[i];
    }
  }
  double cond = std::sqrt(rf) * std::sqrt(rif);
  return std::isfinite(cond) ? cond : Inf;
}

} // namespace

void granger(int rows, int colsx, const double * x, int colsy,
             const double * y, int lag, double * fStats, double * pValues,
             int * unresolved)
{
  const bool self = y == NULL || y == x;
  if (self) { y = x; colsy = colsx; }
  if (lag < 1 || lag > MAXLAG)
    error("gputools: lag must be between 1 and %d", MAXLAG);
  const int T = rows - lag, m = lag + 1, df2 = T - 2 * lag - 1, p = colsx,
    q = colsy, pL = p * lag;
  if (df2 < 1)
    error("gputools: the series are too short for lag %d: need at least %d "
          "times", lag, 3 * lag + 2);
  const Precision prec = Precision::Double;
  const int threads = 256;

  // the scaled series' lags and the effects' targets, once, on the device
  std::vector<double> xs, ys, hL, hLY, hTarget((size_t) T * q);
  scale(rows, p, x, xs);
  lagged(rows, p, lag, xs, hL);
  if (self) { hLY = hL; } else { scale(rows, q, y, ys); lagged(rows, q, lag, ys, hLY); }
  const std::vector<double> & yscaled = self ? xs : ys;
  for (int j = 0; j < q; j++)
    for (int t = 0; t < T; t++)
      hTarget[(size_t) j * T + t] = yscaled[(size_t) j * rows + lag + t];
  double * dL = upload(hL), * dLY = self ? dL : upload(hLY),
    * dTarget = upload(hTarget);
  // each cause's lag columns' squared lengths and cross-products L_i'L_i
  double * dL2 = gpuAlloc<double>(pL), * dLL = gpuAlloc<double>((size_t) p * lag * lag);
  {
    int ncol = pL, TT = T;
    void * args[] = { &TT, &ncol, &dL, &dL2 };
    cudaLaunch("grangerColumnSq", args, gridFor(pL, threads), dim3(threads), prec);
    blas::gemmStridedBatched('T', 'N', lag, lag, T, 1.0, dL, T, (long long) lag * T,
                             dL, T, (long long) lag * T, 0.0, dLL, lag,
                             (long long) lag * lag, p);
  }

  // effects a tile at a time: the fast form holds about six T x m per
  // effect (design, Q, intermediates) and C and G
  size_t perEffect = (size_t) 6 * T * m + (size_t) pL * (m + lag);
  int tile = (int) std::max<size_t>(1, std::min<size_t>(q, ((size_t) 1 << 24) / perEffect));
  double * dD = gpuAlloc<double>((size_t) T * tile * m),
    * dQ = gpuAlloc<double>((size_t) T * tile * m),
    * dGram = gpuAlloc<double>((size_t) tile * m * m),
    * dRinv = gpuAlloc<double>((size_t) tile * m * m),
    * dEffectCond = gpuAlloc<double>(tile),
    * dE = gpuAlloc<double>((size_t) T * tile),
    * dQy = gpuAlloc<double>((size_t) tile * m),
    * dQe = gpuAlloc<double>((size_t) tile * m),
    * dRss = gpuAlloc<double>(tile),
    * dC = gpuAlloc<double>((size_t) tile * m * pL),
    * dLe = gpuAlloc<double>((size_t) pL * tile),
    * dGain = gpuAlloc<double>((size_t) tile * p),
    * dCond = gpuAlloc<double>((size_t) tile * p),
    * dLeft = gpuAlloc<double>((size_t) tile * p);
  int * dBad = gpuAlloc<int>(tile), * dRedo = gpuAlloc<int>((size_t) tile * p),
    * dIdxJ = gpuAlloc<int>((size_t) tile * p), * dIdxI = gpuAlloc<int>((size_t) tile * p);

  std::vector<double> gain((size_t) p * q), causeCond((size_t) p * q),
    left((size_t) p * q), rssR(q), effectCond(q);
  std::vector<double> hGain((size_t) tile * p), hCond((size_t) tile * p),
    hLeft((size_t) tile * p), hEffCond(tile), hRss(tile);
  std::vector<int> hBad(tile), hRedo((size_t) tile * p), hIdxJ, hIdxI;

  for (int j0 = 0; j0 < q; j0 += tile) {
    int b = std::min(tile, q - j0);
    size_t bm = (size_t) b * m;
    // the restricted designs, orthonormalised by Cholesky-QR2
    {
      int TT = T, lg = lag, jj0 = j0, bb = b;
      void * args[] = { &TT, &lg, &jj0, &bb, &dLY, &dD };
      cudaLaunch("grangerDesign", args, gridFor((size_t) T * bm, threads), dim3(threads), prec);
    }
    CUDART_CALL(cudaMemcpy(dQ, dD, (size_t) T * bm * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDART_CALL(cudaMemset(dBad, 0, b * sizeof(int)));
    for (int pass = 0; pass < 2; pass++) {
      blas::gemmStridedBatched('T', 'N', m, m, T, 1.0, dQ, T, (long long) m * T,
                               dQ, T, (long long) m * T, 0.0, dGram, m,
                               (long long) m * m, b);
      int bb = b, mm = m, writeCond = pass == 0;
      double limit = CHOLESKY_QR_LIMIT;
      void * fargs[] = { &bb, &mm, &dGram, &limit, &writeCond, &dRinv, &dEffectCond, &dBad };
      cudaLaunch("grangerFactor", fargs, gridFor(b, threads), dim3(threads), prec);
      int TT = T;
      void * aargs[] = { &TT, &mm, &bb, &dRinv, &dQ };
      cudaLaunch("grangerApplyRinv", aargs, gridFor((size_t) T * b, threads), dim3(threads), prec);
    }
    // Householder on the host for the designs Cholesky-QR2 could not take
    download(dBad, hBad);
    hBad.resize(b); download(dEffectCond, hEffCond); hEffCond.resize(b);
    for (int jj = 0; jj < b; jj++) {
      if (!hBad[jj]) continue;
      std::vector<double> d((size_t) T * m), qh;
      CUDART_CALL(cudaMemcpy(d.data(), dD + (size_t) jj * m * T, d.size() * sizeof(double),
                             cudaMemcpyDeviceToHost));
      hEffCond[jj] = householder(T, m, d, qh);
      CUDART_CALL(cudaMemcpy(dQ + (size_t) jj * m * T, qh.data(), qh.size() * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
    hBad.assign(tile, 0); hEffCond.resize(tile);
    std::copy(hEffCond.begin(), hEffCond.begin() + b, effectCond.begin() + j0);
    // residuals e = y - Q (Q'y), their sums of squares
    CUDART_CALL(cudaMemcpy(dE, dTarget + (size_t) j0 * T, (size_t) T * b * sizeof(double),
                           cudaMemcpyDeviceToDevice));
    blas::gemmStridedBatched('T', 'N', m, 1, T, 1.0, dQ, T, (long long) m * T,
                             dE, T, (long long) T, 0.0, dQy, m, (long long) m, b);
    blas::gemmStridedBatched('N', 'N', T, 1, m, -1.0, dQ, T, (long long) m * T,
                             dQy, m, (long long) m, 1.0, dE, T, (long long) T, b);
    {
      int TT = T, bb = b;
      void * args[] = { &TT, &bb, &dE, &dRss };
      cudaLaunch("grangerRss", args, gridFor(b, threads), dim3(threads), prec);
    }
    // the fast form's ingredients: C = Q'L, L'e, Q'e
    blas::gemm('T', 'N', (int) bm, pL, T, 1.0, dQ, T, dL, T, 0.0, dC, (int) bm);
    blas::gemm('T', 'N', pL, b, T, 1.0, dL, T, dE, T, 0.0, dLe, pL);
    blas::gemmStridedBatched('T', 'N', m, 1, T, 1.0, dQ, T, (long long) m * T,
                             dE, T, (long long) T, 0.0, dQe, m, (long long) m, b);
    {
      int bb = b, pp = p, lg = lag, jj0 = j0, sf = self;
      double limit = FAST_SCORE_LIMIT;
      void * args[] = { &bb, &pp, &lg, &jj0, &sf, &dLL, &dC, &dLe, &dQe, &dL2, &dRss,
                        &limit, &dGain, &dCond, &dLeft, &dRedo };
      cudaLaunch("grangerFast", args, gridFor((size_t) b * p, threads), dim3(threads), prec);
    }
    // the pairs the fast form cannot be trusted on, explicitly
    hRedo.resize((size_t) b * p); download(dRedo, hRedo);
    hIdxJ.clear(); hIdxI.clear();
    for (size_t e = 0; e < (size_t) b * p; e++)
      if (hRedo[e]) { hIdxJ.push_back((int) (e / p)); hIdxI.push_back((int) (e % p)); }
    hRedo.resize((size_t) tile * p);
    int nred = (int) hIdxJ.size();
    if (nred) {
      CUDART_CALL(cudaMemcpy(dIdxJ, hIdxJ.data(), nred * sizeof(int), cudaMemcpyHostToDevice));
      CUDART_CALL(cudaMemcpy(dIdxI, hIdxI.data(), nred * sizeof(int), cudaMemcpyHostToDevice));
      int TT = T, pp = p, lg = lag;
      void * args[] = { &TT, &pp, &lg, &nred, &dIdxJ, &dIdxI, &dQ, &dL, &dE, &dL2,
                        &dGain, &dCond, &dLeft };
      cudaLaunch("grangerExplicit", args, gridFor(nred, threads), dim3(threads), prec);
    }
    hGain.resize((size_t) b * p); download(dGain, hGain);
    hCond.resize((size_t) b * p); download(dCond, hCond);
    hLeft.resize((size_t) b * p); download(dLeft, hLeft);
    hRss.resize(b); download(dRss, hRss);
    for (int jj = 0; jj < b; jj++) {
      rssR[j0 + jj] = hRss[jj];
      for (int i = 0; i < p; i++) {
        size_t src = (size_t) jj * p + i, dst = (size_t) (j0 + jj) * p + i;
        gain[dst] = hGain[src]; causeCond[dst] = hCond[src]; left[dst] = hLeft[src];
      }
    }
    hGain.resize((size_t) tile * p); hCond.resize((size_t) tile * p);
    hLeft.resize((size_t) tile * p); hRss.resize(tile);
  }
  // free before the error check: it raises (longjmps) if a kernel faulted
  cudaFree(dD); cudaFree(dQ); cudaFree(dGram); cudaFree(dRinv); cudaFree(dEffectCond);
  cudaFree(dE); cudaFree(dQy); cudaFree(dQe); cudaFree(dRss); cudaFree(dC); cudaFree(dLe);
  cudaFree(dGain); cudaFree(dCond); cudaFree(dLeft); cudaFree(dBad); cudaFree(dRedo);
  cudaFree(dIdxJ); cudaFree(dIdxI); cudaFree(dL2); cudaFree(dLL);
  if (!self) cudaFree(dLY);
  cudaFree(dL); cudaFree(dTarget);
  if (hasCudaError("granger")) { /* the error is raised by the check */ }

  // the tests, NaN where unresolvable: a series on itself and a constant
  // series have no test (NaN, not counted); the rest are counted for the
  // caller's warning
  int count = 0;
  for (int j = 0; j < q; j++) {
    bool effectWeak = std::isfinite(effectCond[j]) && effectCond[j] > EFFECT_LIMIT;
    for (int i = 0; i < p; i++) {
      size_t e = (size_t) j * p + i;
      double gn = gain[e];
      bool weak = (effectWeak || causeCond[e] > CAUSE_LIMIT || left[e] < LEFT_LIMIT)
        && (std::isfinite(gn) || std::isinf(causeCond[e]));
      if (self && i == j) weak = false;
      if (weak) { gn = NaN; count++; }
      if (std::isnan(gn)) { fStats[e] = NaN; pValues[e] = NaN; continue; }
      // rounding can take a gain of 0 just below 0, or one equal to the
      // restricted RSS just above it
      gn = std::min(std::max(gn, 0.0), rssR[j]);
      double rssU = rssR[j] - gn;
      double f = (gn / lag) / (rssU / df2);
      fStats[e] = f;
      // the upper tail directly: 1 - pf(lower tail) rounds small p to 0
      pValues[e] = pf(f, (double) lag, (double) df2, 0, 0);
    }
  }
  *unresolved = count;
}
