// Granger causality F tests for every ordered pair of series, by the
// Frisch-Waugh route the host code in src/granger.cpp drives:
//
//   - each effect's restricted design D = [1, own lags] (T x m, m = lag + 1)
//     is orthonormalised to Q by Cholesky-QR2 (the kernels here factor the
//     small Gram matrices and apply R^{-1}; the Gram products and the big
//     products below are cuBLAS calls from the host);
//   - with e the effect's residual and L every cause's lags (T x p*lag),
//     the gain RSS_r - RSS_u of a pair is g' G^{-1} g for the cause's lags
//     with the effect's own past projected out, M = L_i - Q Q' L_i: G = M'M,
//     g = M'e;
//   - the fast form computes those from the small products C = Q'L, L'e and
//     Q'e as G = L'L - C'C and g = L'e - C'(Q'e), and a per-pair score says
//     whether that cancellation can be trusted; the pairs it cannot be
//     trusted on are recomputed from M formed explicitly.
//
// Everything is in double (the host always launches the double build);
// lag is at most GRANGER_MAXLAG so the per-pair systems fit in registers.
// All matrices are column-major, as cuBLAS has them.

#define GRANGER_MAXLAG 16
#define GRANGER_MAXM (GRANGER_MAXLAG + 1)

// Cholesky of the symmetric n x n matrix a (column-major, upper triangle
// read) into the upper factor r (a = r' r). Returns 0, or k > 0 if pivot k
// was not positive (or not finite).
__device__ int grangerCholesky(int n, const real * a, int lda, real * r)
{
  for (int j = 0; j < n; j++) {
    for (int i = 0; i <= j; i++) {
      real s = a[i + j * lda];
      for (int k = 0; k < i; k++) s -= r[k + i * n] * r[k + j * n];
      if (i == j) {
        if (!(s > real(0)) || !isfinite(s)) return j + 1;
        r[j + j * n] = sqrt(s);
      } else {
        r[i + j * n] = s / r[i + i * n];
      }
    }
    for (int i = j + 1; i < n; i++) r[i + j * n] = real(0);
  }
  return 0;
}

// Inverse of the upper triangular n x n matrix r into ri.
__device__ void grangerInvertUpper(int n, const real * r, real * ri)
{
  for (int j = 0; j < n; j++) {
    for (int i = n - 1; i >= 0; i--) {
      if (i > j) { ri[i + j * n] = real(0); continue; }
      real s = (i == j) ? real(1) : real(0);
      for (int k = i + 1; k <= j; k++) s -= r[i + k * n] * ri[k + j * n];
      ri[i + j * n] = s / r[i + i * n];
    }
  }
}

// ||r||_F ||r^{-1}||_F: at least the 2-norm condition number of r and at
// most n times it.
__device__ real grangerCondUpper(int n, const real * r, const real * ri)
{
  real a = real(0), b = real(0);
  for (int j = 0; j < n; j++)
    for (int i = 0; i <= j; i++) {
      a += r[i + j * n] * r[i + j * n];
      b += ri[i + j * n] * ri[i + j * n];
    }
  return sqrt(a) * sqrt(b);
}

// The restricted design of each of b effects, side by side: D (T x b*m),
// effect jj's columns jj*m .. jj*m + m - 1 holding the intercept at unit
// length and its lag columns from ly (T x q*lag; effect j0 + jj's are
// (j0 + jj)*lag .. + lag - 1).
__global__ void grangerDesign(int T, int lag, int j0, int b, const real * ly,
                              real * d)
{
  int m = lag + 1;
  size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (e >= (size_t) T * b * m) return;
  int t = e % T, col = e / T, jj = col / m, a = col % m;
  d[e] = a == 0 ? rsqrt((real) T)
                : ly[t + ((size_t) (j0 + jj) * lag + a - 1) * T];
}

// From each effect's m x m Gram matrix g (= D'D, b of them): the inverse
// of its upper Cholesky factor into rinv, the factor's condition estimate
// into cond (on the first pass only, when writeCond is set), and bad set
// where the factorisation failed or the estimate is not below limit.
__global__ void grangerFactor(int b, int m, const real * g, real limit,
                              int writeCond, real * rinv, real * cond,
                              int * bad)
{
  int jj = blockIdx.x * blockDim.x + threadIdx.x;
  if (jj >= b) return;
  real r[GRANGER_MAXM * GRANGER_MAXM], ri[GRANGER_MAXM * GRANGER_MAXM];
  int info = grangerCholesky(m, g + (size_t) jj * m * m, m, r);
  real c;
  if (info) {
    c = REAL_INF;
    for (int k = 0; k < m * m; k++) ri[k] = REAL_NAN;
  } else {
    grangerInvertUpper(m, r, ri);
    c = grangerCondUpper(m, r, ri);
  }
  if (writeCond) cond[jj] = c;
  if (info || !(c < limit)) bad[jj] = 1;
  for (int k = 0; k < m * m; k++) rinv[(size_t) jj * m * m + k] = ri[k];
}

// Q = Q R^{-1} in place, each thread one row of one effect's T x m block.
__global__ void grangerApplyRinv(int T, int m, int b, const real * rinv,
                                 real * q)
{
  size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (e >= (size_t) T * b) return;
  int t = e % T, jj = e / T;
  real * row = q + (size_t) jj * m * T + t;
  const real * ri = rinv + (size_t) jj * m * m;
  real v[GRANGER_MAXM];
  for (int a = 0; a < m; a++) v[a] = row[(size_t) a * T];
  for (int c = 0; c < m; c++) {
    real s = real(0);
    for (int a = 0; a <= c; a++) s += v[a] * ri[a + c * m];
    row[(size_t) c * T] = s;
  }
}

// Each effect's residual sum of squares from its residual column of e.
__global__ void grangerRss(int T, int b, const real * e, real * rss)
{
  int jj = blockIdx.x * blockDim.x + threadIdx.x;
  if (jj >= b) return;
  const real * col = e + (size_t) jj * T;
  real s = real(0);
  for (int t = 0; t < T; t++) s += col[t] * col[t];
  rss[jj] = s;
}

// Each cause's lag columns' squared lengths: l2 (p*lag).
__global__ void grangerColumnSq(int T, int ncol, const real * l, real * l2)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncol) return;
  const real * col = l + (size_t) c * T;
  real s = real(0);
  for (int t = 0; t < T; t++) s += col[t] * col[t];
  l2[c] = s;
}

// Solves G x = g for a pair (G lag x lag, symmetric positive definite if
// the pair is resolvable) and returns the gain g'x, the Frobenius
// estimate of G's condition number (infinite if the factorisation fails,
// NaN if G is not finite) and the smallest diagonal fraction left.
__device__ real grangerSolvePair(int lag, const real * G, const real * g,
                                 const real * l2i, real * cond, real * left)
{
  real r[GRANGER_MAXLAG * GRANGER_MAXLAG], ri[GRANGER_MAXLAG * GRANGER_MAXLAG];
  real lft = REAL_INF;
  int finite = 1;
  for (int k = 0; k < lag * lag; k++) if (!isfinite(G[k])) finite = 0;
  for (int k = 0; k < lag; k++) {
    real f = G[k + k * lag] / l2i[k];
    if (isnan(f)) { lft = f; break; }   // a NaN propagates
    if (f < lft) lft = f;
  }
  *left = lft;
  if (!finite) { *cond = REAL_NAN; return REAL_NAN; }
  if (grangerCholesky(lag, G, lag, r)) { *cond = REAL_INF; return REAL_NAN; }
  grangerInvertUpper(lag, r, ri);
  // ||G||_F ||G^{-1}||_F, G^{-1} = R^{-1} R^{-T}
  real a = real(0), bsum = real(0);
  for (int k = 0; k < lag * lag; k++) a += G[k] * G[k];
  for (int i = 0; i < lag; i++)
    for (int j = 0; j < lag; j++) {
      real s = real(0);
      for (int k = (i > j ? i : j); k < lag; k++) s += ri[i + k * lag] * ri[j + k * lag];
      bsum += s * s;
    }
  *cond = sqrt(a) * sqrt(bsum);
  // x = R^{-1} R^{-T} g, gain = g' x
  real y[GRANGER_MAXLAG], gain = real(0);
  for (int i = 0; i < lag; i++) {            // R' y = g (forward)
    real s = g[i];
    for (int k = 0; k < i; k++) s -= r[k + i * lag] * y[k];
    y[i] = s / r[i + i * lag];
  }
  for (int i = 0; i < lag; i++) gain += y[i] * y[i];   // g' G^{-1} g = y' y
  return gain;
}

// The fast form for every (effect jj of the tile, cause i): from
// ll (lag x lag per cause), c = Q'L (b*m x p*lag), le = L'e (p*lag x b),
// qe = Q'e (m x b), l2 and rss. Writes gain, cond, left (b x p, cause
// fastest) and redo = 1 where the pair must be recomputed explicitly: its
// score cond / left * rss / (rss - gain) is not below limit (a NaN score,
// a constant series, has no test either way). Pairs of a series with
// itself (self set, cause i == effect j0 + jj) are NaN with no recompute.
__global__ void grangerFast(int b, int p, int lag, int j0, int self,
                            const real * ll, const real * c, const real * le,
                            const real * qe, const real * l2, const real * rss,
                            real limit, real * gain, real * cond, real * left,
                            int * redo)
{
  size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (e >= (size_t) b * p) return;
  int i = e % p, jj = e / p, m = lag + 1, ldc = b * m;
  if (self && i == j0 + jj) {
    gain[e] = REAL_NAN; cond[e] = REAL_NAN; left[e] = REAL_NAN; redo[e] = 0;
    return;
  }
  real G[GRANGER_MAXLAG * GRANGER_MAXLAG], g[GRANGER_MAXLAG];
  const real * cb = c + (size_t) jj * m + (size_t) i * lag * ldc;  // C[a, k] at cb[a + k*ldc]
  const real * qeb = qe + (size_t) jj * m;
  for (int k = 0; k < lag; k++) {
    for (int l = 0; l < lag; l++) {
      real s = ll[k + l * lag + (size_t) i * lag * lag];
      for (int a = 0; a < m; a++) s -= cb[a + (size_t) k * ldc] * cb[a + (size_t) l * ldc];
      G[k + l * lag] = s;
    }
    real s = le[(size_t) i * lag + k + (size_t) jj * (p * lag)];
    for (int a = 0; a < m; a++) s -= cb[a + (size_t) k * ldc] * qeb[a];
    g[k] = s;
  }
  real cnd, lft;
  real gn = grangerSolvePair(lag, G, g, l2 + (size_t) i * lag, &cnd, &lft);
  gain[e] = gn; cond[e] = cnd; left[e] = lft;
  real r = rss[jj], amp = r / (r - (gn > real(0) ? gn : real(0)));
  if (!(amp > real(0))) amp = REAL_INF;
  real score = lft > real(0) ? cnd / lft * amp : REAL_INF;
  redo[e] = (!(score < limit) && !isnan(score)) ? 1 : 0;
}

// The explicit form for nred flagged pairs (effect idxJ[r] of the tile,
// cause idxI[r]): M = L_i - Q_jj (Q_jj' L_i) formed in two passes over the
// T rows, then G = M'M and g = M'e, and the same solve. One thread per
// pair.
__global__ void grangerExplicit(int T, int p, int lag, int nred,
                                const int * idxJ, const int * idxI,
                                const real * q, const real * l, const real * e,
                                const real * l2, real * gain, real * cond,
                                real * left)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nred) return;
  int jj = idxJ[r], i = idxI[r], m = lag + 1;
  const real * qb = q + (size_t) jj * m * T;
  const real * lb = l + (size_t) i * lag * T;
  const real * eb = e + (size_t) jj * T;
  real cc[GRANGER_MAXM * GRANGER_MAXLAG];           // Q' L_i, m x lag
  for (int k = 0; k < m * lag; k++) cc[k] = real(0);
  for (int t = 0; t < T; t++)
    for (int a = 0; a < m; a++) {
      real qa = qb[t + (size_t) a * T];
      for (int k = 0; k < lag; k++) cc[a + k * m] += qa * lb[t + (size_t) k * T];
    }
  real G[GRANGER_MAXLAG * GRANGER_MAXLAG], g[GRANGER_MAXLAG], mk[GRANGER_MAXLAG];
  for (int k = 0; k < lag * lag; k++) G[k] = real(0);
  for (int k = 0; k < lag; k++) g[k] = real(0);
  for (int t = 0; t < T; t++) {
    for (int k = 0; k < lag; k++) {
      real s = lb[t + (size_t) k * T];
      for (int a = 0; a < m; a++) s -= qb[t + (size_t) a * T] * cc[a + k * m];
      mk[k] = s;
    }
    real et = eb[t];
    for (int k = 0; k < lag; k++) {
      g[k] += mk[k] * et;
      for (int l = 0; l < lag; l++) G[k + l * lag] += mk[k] * mk[l];
    }
  }
  real cnd, lft;
  size_t o = (size_t) jj * p + i;
  gain[o] = grangerSolvePair(lag, G, g, l2 + (size_t) i * lag, &cnd, &lft);
  cond[o] = cnd; left[o] = lft;
}
