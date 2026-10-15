#include <stdio.h>
#include <string.h>
#include <vector>
#include <cmath>

#include "cuda_runtime.h"

#include "R.h"
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

#include "correlation.h"
#include "cuseful.h"
#include "distance.h"
#include "granger.h"
#include "hcluster.h"
#include "kendall.h"
#include "lsfit.h"
#include "matmult.h"
#include "mi.h"
#include "qrdecomp.h"
#include "solve.h"
#include "rank.h"
#include "pairs.h"
#include "kmeans.h"

#include "rinterface.h"
#include "routines.h"
#include "cudaUtils.h"
#include "gpublas.h"

// The .Call entry points. The kernels work in single precision: inputs
// arrive as R doubles and are narrowed here, results are widened back to
// doubles, so R never sees a "Csingle" object.

namespace {

// Converts a numeric vector to T (narrowing to float for single
// precision). Unless naok, refuses NA, NaN and Inf the way .C() used to.
template <typename T>
std::vector<T> narrow(SEXP x, const char * what, bool naok = false)
{
  if (!isNumeric(x) && !isLogical(x))
    error("gputools: %s must be numeric", what);
  x = PROTECT(coerceVector(x, REALSXP));
  R_xlen_t n = XLENGTH(x);
  const double * px = REAL(x);
  std::vector<T> out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    if (!naok && !R_FINITE(px[i]))
      error("gputools: NA/NaN/Inf in %s", what);
    out[i] = (T) px[i];
  }
  UNPROTECT(1);
  return out;
}

// A new double vector holding v.
template <typename T>
SEXP doubles(const T * v, R_xlen_t n)
{
  SEXP out = allocVector(REALSXP, n);
  double * po = REAL(out);
  for (R_xlen_t i = 0; i < n; ++i) po[i] = v[i];
  return out;
}

// A double copy of like (dims and all) overwritten with v.
template <typename T>
SEXP doublesLike(SEXP like, const T * v)
{
  // coerceVector keeps the attributes; duplicate in case it returned like
  SEXP out = PROTECT(duplicate(coerceVector(like, REALSXP)));
  double * po = REAL(out);
  for (R_xlen_t i = 0; i < XLENGTH(out); ++i) po[i] = v[i];
  UNPROTECT(1);
  return out;
}

// names must end with "".
SEXP namedList(const char ** names, SEXP * values)
{
  SEXP out = PROTECT(mkNamed(VECSXP, names));
  for (int i = 0; names[i][0]; ++i) SET_VECTOR_ELT(out, i, values[i]);
  UNPROTECT(1);
  return out;
}

const char * string(SEXP x)
{
  return CHAR(STRING_ELT(x, 0));
}

// The precision argument every entry point takes: "single" or "double".
bool isDouble(SEXP precision)
{
  const char * p = string(precision);
  if (strcmp(p, "double") == 0) return true;
  if (strcmp(p, "single") == 0) return false;
  error("gputools: precision must be \"single\" or \"double\"");
}

} // namespace

// The names the R functions pass. An unknown name is an error: falling
// back to a default once ran "flexible group" as single linkage.
dist_method getDistEnum(const char * methodStr)
{
  if(0 == strcmp(methodStr,"euclidean"))  return EUCLIDEAN;
  if(0 == strcmp(methodStr,"maximum"))    return MAXIMUM;
  if(0 == strcmp(methodStr,"manhattan"))  return MANHATTAN;
  if(0 == strcmp(methodStr,"canberra"))   return CANBERRA;
  if(0 == strcmp(methodStr,"binary"))     return BINARY;
  if(0 == strcmp(methodStr,"minkowski"))  return MINKOWSKI;
  error("gputools: unknown distance method \"%s\"", methodStr);
}

hc_method getClusterEnum(const char * methodStr)
{
  if(0 == strcmp(methodStr,"single"))         return SINGLE;
  if(0 == strcmp(methodStr,"complete"))       return COMPLETE;
  if(0 == strcmp(methodStr,"wpgma"))          return WPGMA;
  if(0 == strcmp(methodStr,"average"))        return AVERAGE;
  if(0 == strcmp(methodStr,"median"))         return MEDIAN;
  if(0 == strcmp(methodStr,"centroid"))       return CENTROID;
  if(0 == strcmp(methodStr,"flexible group")) return FLEXIBLE_GROUP;
  if(0 == strcmp(methodStr,"flexible"))       return FLEXIBLE;
  if(0 == strcmp(methodStr,"ward"))           return WARD;
  if(0 == strcmp(methodStr,"mcquitty"))       return MCQUITTY;
  error("gputools: unknown clustering method \"%s\"", methodStr);
}

void formatClustering(const int len, const int * sub,  const int * sup, 
                      int * merge, int * order)
{
  for(size_t i = 0; i < len; i++) {
    merge[i] = -(sub[i] + 1);
    merge[i+len] = -(sup[i] + 1);
  }

  for(size_t i = 0; i < len; i++) {
    for(size_t j = i+1; j < len; j++) {
      if((merge[j] == merge[i]) || (merge[j] == merge[i+len]))
        merge[j] = i + 1;
      if((merge[j+len] == merge[i]) || (merge[j+len] == merge[i+len]))
        merge[j+len] = i + 1;
      if(((merge[j+len] < 0) && (merge[j] > 0)) 
         || ((merge[j] > 0) && (merge[j+len] > 0) 
             && (merge[j] > merge[j+len]))) {
        int holder = merge[j];
        merge[j] = merge[j+len];
        merge[j+len] = holder; 
      }
    }
  }
  getPrintOrder(len, merge, order);
}

void getPrintOrder(const int len, const int * merge, int * order)
{
  int 
    level = len-1, otop = len;

  depthFirst(len, merge, level, &otop, order);
}

void depthFirst(const int len, const int * merge, int level, int * otop, 
                int * order)
{
  int
    left = level, right = level + len;

  if(merge[right] < 0) {
    order[*otop] = -merge[right];
    (*otop)--;
  } else
    depthFirst(len, merge, merge[right]-1, otop, order);

  if(merge[left] < 0) {
    order[*otop] = -merge[left];
    (*otop)--;
  } else
    depthFirst(len, merge, merge[left]-1, otop, order);
}

namespace {

// Standardises each column of the n x p matrix x so that the correlation
// of two columns is the dot product of their standardised versions:
// z = (x - mean) / (sd sqrt(n - 1)), mean and sd taken in double. A column
// with a missing value, or a constant one, becomes NaN, so its
// correlations come out NA, as with cor(). norms, if given, gets each
// centred column's length.
template <typename T>
std::vector<T> standardised(SEXP x, int n, int p,
                            std::vector<double> * norms = NULL)
{
  x = PROTECT(coerceVector(x, REALSXP));
  const double * px = REAL(x);
  std::vector<T> z((size_t) n * p);
  for (int j = 0; j < p; ++j) {
    const double * col = px + (size_t) j * n;
    double mean = 0, ss = 0;
    for (int i = 0; i < n; ++i) mean += col[i];
    mean /= n;
    for (int i = 0; i < n; ++i) ss += (col[i] - mean) * (col[i] - mean);
    double scale = 1 / std::sqrt(ss);   // = 1 / (sd sqrt(n - 1))
    if (norms) norms->push_back(std::sqrt(ss));
    for (int i = 0; i < n; ++i)
      z[i + (size_t) j * n] = (T) ((col[i] - mean) * scale);
  }
  UNPROTECT(1);
  return z;
}

// Pearson correlations of every column of x with every column of y (x
// itself when y is NULL) over all rows, as one matrix product on the GPU:
// p x q, column-major.
template <typename T>
SEXP corProductT(SEXP x, SEXP y)
{
  bool same = isNull(y);   // y = NULL: x with itself
  SEXP dx = getAttrib(x, R_DimSymbol);
  int n = INTEGER(dx)[0], p = INTEGER(dx)[1],
    q = same ? p : INTEGER(getAttrib(y, R_DimSymbol))[1];
  std::vector<T> zx = standardised<T>(x, n, p), zy;
  if (!same) zy = standardised<T>(y, n, q);

  T * dZx = gpuAlloc<T>((size_t) n * p), * dZy = same ? dZx
    : gpuAlloc<T>((size_t) n * q), * dR = gpuAlloc<T>((size_t) p * q);
  blas::setMatrix(n, p, sizeof(T), zx.data(), n, dZx, n);
  if (!same) blas::setMatrix(n, q, sizeof(T), zy.data(), n, dZy, n);
  blas::gemm('T', 'N', p, q, n, T(1), dZx, n, dZy, n, T(0), dR, p);

  std::vector<T> r((size_t) p * q);
  blas::getMatrix(p, q, sizeof(T), dR, p, r.data(), p);
  cudaFree(dZx);
  if (!same) cudaFree(dZy);
  cudaFree(dR);

  // list(corr, ts): the correlations and their t statistics on n - 2
  // degrees of freedom (as tScore() in correlation.cpp), both p x q
  const char * names[] = { "corr", "ts", "" };
  SEXP values[] = {
    PROTECT(allocMatrix(REALSXP, p, q)), PROTECT(allocMatrix(REALSXP, p, q))
  };
  double * pc = REAL(values[0]), * pt = REAL(values[1]);
  for (size_t i = 0; i < r.size(); ++i) {
    double v = r[i];
    if (v > 1) v = 1;
    if (v < -1) v = -1;
    pc[i] = std::isnan(v) ? NA_REAL : v;
    double denom = 1 - v * v;
    pt[i] = std::isnan(v) ? NA_REAL
      : n < 3 ? R_NaN
      : denom > 0 ? v * std::sqrt((n - 2) / denom)
      : (v > 0 ? R_PosInf : R_NegInf);
  }
  SEXP out = namedList(names, values);
  UNPROTECT(2);
  return out;
}

// list(i, j, r) from pairs, i and j 1-based, and if given the columns'
// residual lengths as xlen, ylen
SEXP pairListValue(const PairList & pairs, const std::vector<double> * xLen,
                   const std::vector<double> * yLen)
{
  R_xlen_t m = pairs.r.size();
  const char * names[] = { "i", "j", "r", "xlen", "ylen", "" };
  if (!xLen) names[3] = "";
  SEXP values[] = {
    PROTECT(allocVector(INTSXP, m)), PROTECT(allocVector(INTSXP, m)),
    PROTECT(allocVector(REALSXP, m)),
    PROTECT(allocVector(REALSXP, xLen ? xLen->size() : 0)),
    PROTECT(allocVector(REALSXP, yLen ? yLen->size() : 0))
  };
  int * pi = INTEGER(values[0]), * pj = INTEGER(values[1]);
  double * pr = REAL(values[2]);
  for (R_xlen_t e = 0; e < m; ++e) {
    pi[e] = pairs.i[e] + 1;
    pj[e] = pairs.j[e] + 1;
    double v = pairs.r[e];
    pr[e] = v > 1 ? 1 : (v < -1 ? -1 : v);
  }
  if (xLen) std::copy(xLen->begin(), xLen->end(), REAL(values[3]));
  if (yLen) std::copy(yLen->begin(), yLen->end(), REAL(values[4]));
  SEXP out = namedList(names, values);
  UNPROTECT(5);
  return out;
}

// Selected Pearson correlations of the columns of x with those of y (x
// itself when y is NULL): every pair with |r| >= threshold (mode 0), or
// each y column's top k partners (mode 1). list(i, j, r), i and j 1-based.
template <typename T>
SEXP corPairsT(SEXP x, SEXP y, int mode, double threshold, int k,
               bool absolute, int tile, double limit)
{
  bool same = isNull(y);
  SEXP dx = getAttrib(x, R_DimSymbol);
  int n = INTEGER(dx)[0], p = INTEGER(dx)[1],
    q = same ? p : INTEGER(getAttrib(y, R_DimSymbol))[1];
  std::vector<T> zx = standardised<T>(x, n, p), zy;
  if (!same) zy = standardised<T>(y, n, q);
  T * dZx = gpuAlloc<T>((size_t) n * p), * dZy = dZx;
  CUDART_CALL(cudaMemcpy(dZx, zx.data(), zx.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
  if (!same) {
    dZy = gpuAlloc<T>((size_t) n * q);
    CUDART_CALL(cudaMemcpy(dZy, zy.data(), zy.size() * sizeof(T),
                           cudaMemcpyHostToDevice));
  }

  PairList pairs;
  selectPairs<T>(n, p, dZx, q, dZy, mode, threshold, k, absolute, tile,
                 limit, pairs);
  return pairListValue(pairs, NULL, NULL);
}

// Association of each column of x with each column of y, adjusted for the
// covariates: their correlation after both are residualised on the m
// orthonormal columns of q (an intercept and the covariates). The pairs
// selected as by corPairsT, with each column's residual length (NaN where
// nothing is left) for the regression slopes.
template <typename T>
SEXP assocT(SEXP x, SEXP y, SEXP q, int mode, double threshold, int k,
            int tile, double limit)
{
  SEXP dx = getAttrib(x, R_DimSymbol), dq = getAttrib(q, R_DimSymbol);
  int n = INTEGER(dx)[0], p = INTEGER(dx)[1],
    nq = INTEGER(getAttrib(y, R_DimSymbol))[1], m = INTEGER(dq)[1];
  if (INTEGER(dq)[0] != n || INTEGER(getAttrib(y, R_DimSymbol))[0] != n)
    error("gputools: x, y and the covariate basis must have the same rows");
  std::vector<double> xLen, yLen, xRel, yRel;
  std::vector<T> zx = standardised<T>(x, n, p, &xLen),
    zy = standardised<T>(y, n, nq, &yLen);
  std::vector<T> hq((size_t) n * m);
  const double * pq = REAL(q);
  for (size_t e = 0; e < hq.size(); ++e) hq[e] = (T) pq[e];

  T * dZx = gpuAlloc<T>(zx.size()), * dZy = gpuAlloc<T>(zy.size()),
    * dQ = gpuAlloc<T>(hq.size() > 0 ? hq.size() : 1);
  CUDART_CALL(cudaMemcpy(dZx, zx.data(), zx.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dZy, zy.data(), zy.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dQ, hq.data(), hq.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
  residualise<T>(n, p, dZx, m, dQ, xRel);
  residualise<T>(n, nq, dZy, m, dQ, yRel);
  cudaFree(dQ);
  for (int j = 0; j < p; ++j) xLen[j] *= xRel[j];
  for (int j = 0; j < nq; ++j) yLen[j] *= yRel[j];

  PairList pairs;
  selectPairs<T>(n, p, dZx, nq, dZy, mode, threshold, k, true, tile, limit,
                 pairs);
  return pairListValue(pairs, &xLen, &yLen);
}

// Lloyd's k-means of the rows of x from the starting centres:
// list(cluster, centers, size, iter), as R's kmeans_Lloyd returns them
template <typename T>
SEXP kmeansT(SEXP x, SEXP centers, int maxiter)
{
  SEXP dx = getAttrib(x, R_DimSymbol);
  int n = INTEGER(dx)[0], d = INTEGER(dx)[1],
    k = INTEGER(getAttrib(centers, R_DimSymbol))[0];
  std::vector<T> hx((size_t) n * d);
  const double * px = REAL(x);
  for (size_t e = 0; e < hx.size(); ++e) hx[e] = (T) px[e];

  const char * names[] = { "cluster", "centers", "size", "iter", "" };
  SEXP values[] = {
    PROTECT(allocVector(INTSXP, n)), PROTECT(allocMatrix(REALSXP, k, d)),
    PROTECT(allocVector(INTSXP, k)), PROTECT(allocVector(INTSXP, 1))
  };
  std::copy(REAL(centers), REAL(centers) + (size_t) k * d, REAL(values[1]));
  lloyd<T>(n, d, hx.data(), k, REAL(values[1]), maxiter, INTEGER(values[0]),
           INTEGER(values[2]), INTEGER(values[3]));
  SEXP out = namedList(names, values);
  UNPROTECT(4);
  return out;
}

// rank(ties.method = "average", na.last = "keep") of each column of x,
// on the GPU
template <typename T>
SEXP rankT(SEXP x)
{
  SEXP dx = getAttrib(x, R_DimSymbol);
  int n = INTEGER(dx)[0], p = INTEGER(dx)[1];
  std::vector<T> a = narrow<T>(x, "x", true), r((size_t) n * p);
  rankColumns(a.data(), n, p, r.data());
  SEXP out = PROTECT(allocMatrix(REALSXP, n, p));
  double * po = REAL(out);
  for (size_t i = 0; i < r.size(); ++i)
    po[i] = std::isnan(r[i]) ? NA_REAL : r[i];
  UNPROTECT(1);
  return out;
}

template <typename T>
SEXP pmccT(UseObs obs, SEXP x, int na, SEXP y, int nb, int size)
{
  std::vector<T>
    a = narrow<T>(x, "x", true), b = narrow<T>(y, "y", true),
    pairs(na * nb), corr(na * nb), ts(na * nb);
  pmcc(obs, a.data(), na, b.data(), nb, size,
       pairs.data(), corr.data(), ts.data());

  const char * names[] = { "pairs", "corr", "ts", "" };
  SEXP values[] = {
    PROTECT(doubles(pairs.data(), na * nb)),
    PROTECT(doubles(corr.data(), na * nb)),
    PROTECT(doubles(ts.data(), na * nb))
  };
  SEXP out = namedList(names, values);
  UNPROTECT(3);
  return out;
}

template <typename T>
SEXP ttestT(SEXP goodPairs, SEXP coeffs)
{
  std::vector<T>
    pairs = narrow<T>(goodPairs, "goodPairs", true),
    cors = narrow<T>(coeffs, "coeffs", true),
    ts(pairs.size());
  if (cors.size() != pairs.size())
    error("gputools: goodPairs and coeffs differ in length");
  testSignif(pairs.data(), cors.data(), pairs.size(), ts.data());
  return doubles(ts.data(), ts.size());
}

// Kendall (kernel gpuKendall) or pairwise Spearman (gpuSpearmanPairwise)
// over pairwise complete rows: list(r, n), the correlations and the
// number of complete rows each used
template <typename T>
SEXP rankCorT(const char * kernel, SEXP x, int na, SEXP y, int nb, int size)
{
  std::vector<T> a = narrow<T>(x, "x", true), b = narrow<T>(y, "y", true);
  const char * names[] = { "r", "n", "" };
  SEXP values[] = {
    PROTECT(allocVector(REALSXP, (R_xlen_t) na * nb)),
    PROTECT(allocVector(REALSXP, (R_xlen_t) na * nb))
  };
  rankCorrelation(kernel, a.data(), na, b.data(), nb, size, REAL(values[0]),
                  REAL(values[1]));
  SEXP out = namedList(names, values);
  UNPROTECT(2);
  return out;
}

template <typename T>
SEXP fAndP(const std::vector<T> & f, const std::vector<T> & p)
{
  const char * names[] = { "fStats", "pValues", "" };
  SEXP values[] = {
    PROTECT(doubles(f.data(), f.size())), PROTECT(doubles(p.data(), p.size()))
  };
  SEXP out = namedList(names, values);
  UNPROTECT(2);
  return out;
}

template <typename T>
SEXP grangerT(SEXP x, int rows, int n, int lag)
{
  std::vector<T> a = narrow<T>(x, "x"), f(n * n), p(n * n);
  granger(rows, n, a.data(), lag, f.data(), p.data());
  return fAndP(f, p);
}

template <typename T>
SEXP grangerXYT(SEXP x, int rows, int nx, SEXP y, int ny, int lag)
{
  std::vector<T>
    a = narrow<T>(x, "x"), b = narrow<T>(y, "y"), f(nx * ny), p(nx * ny);
  grangerxy(rows, nx, a.data(), ny, b.data(), lag, f.data(), p.data());
  return fAndP(f, p);
}

template <typename T>
SEXP mutualInfoT(int bins, int order, int nsamples, SEXP x, int na, SEXP y,
                 int nb)
{
  std::vector<T> a = narrow<T>(x, "x"), b = narrow<T>(y, "y"), mi(na * nb);
  bSplineMutualInfo(bins, order, nsamples, na, a.data(), nb, b.data(),
                    mi.data());
  return doubles(mi.data(), mi.size());
}

template <typename T>
SEXP qrDecompT(SEXP x, SEXP tol)
{
  SEXP dim = getAttrib(x, R_DimSymbol);
  int rows = INTEGER(dim)[0], cols = INTEGER(dim)[1];
  std::vector<T> a = narrow<T>(x, "x");

  const char * names[] = { "qr", "pivot", "qraux", "rank", "" };
  SEXP values[] = {
    R_NilValue,
    PROTECT(allocVector(INTSXP, cols)),
    PROTECT(allocVector(REALSXP, cols)),
    PROTECT(allocVector(INTSXP, 1))
  };
  int * pivot = INTEGER(values[1]);
  for (int i = 0; i < cols; ++i) pivot[i] = i;
  memset(REAL(values[2]), 0, cols * sizeof(double));
  INTEGER(values[3])[0] = 0;

  T * dQR = gpuAlloc<T>(a.size());
  CUDART_CALL(cudaMemcpy(dQR, a.data(), a.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
  getQRDecompRR(rows, cols, asReal(tol), dQR, pivot, REAL(values[2]),
                INTEGER(values[3]));
  CUDART_CALL(cudaMemcpy(a.data(), dQR, a.size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
  cudaFree(dQR);

  values[0] = PROTECT(doublesLike(x, a.data()));
  SEXP out = namedList(names, values);
  UNPROTECT(4);
  return out;
}

template <typename T>
SEXP solveT(SEXP x, SEXP y)
{
  SEXP dx = getAttrib(x, R_DimSymbol), dy = getAttrib(y, R_DimSymbol);
  int n = INTEGER(dx)[0], p = INTEGER(dx)[1], k = INTEGER(dy)[1];
  std::vector<T> a = narrow<T>(x, "x"), b = narrow<T>(y, "y"),
    sol((size_t) p * k);
  solveLinear(n, p, a.data(), k, b.data(), sol.data());
  SEXP out = PROTECT(doubles(sol.data(), sol.size()));
  SEXP dim = PROTECT(allocVector(INTSXP, 2));
  INTEGER(dim)[0] = p;
  INTEGER(dim)[1] = k;
  setAttrib(out, R_DimSymbol, dim);
  UNPROTECT(2);
  return out;
}

// svd() of an m x n matrix with m >= n (the R side transposes a wide
// one): list(d, u, v), u and v NULL unless asked for
template <typename T>
SEXP svdT(SEXP x, bool wantU, bool wantV)
{
  SEXP dx = getAttrib(x, R_DimSymbol);
  int m = INTEGER(dx)[0], n = INTEGER(dx)[1];
  std::vector<T> a = narrow<T>(x, "x"), s(n), u(wantU ? (size_t) m * n : 0),
    v(wantV ? (size_t) n * n : 0);
  singularValues(m, n, a.data(), s.data(), wantU, u.data(), wantV, v.data());

  const char * names[] = { "d", "u", "v", "" };
  SEXP values[] = {
    PROTECT(doubles(s.data(), n)),
    wantU ? PROTECT(allocMatrix(REALSXP, m, n)) : R_NilValue,
    wantV ? PROTECT(allocMatrix(REALSXP, n, n)) : R_NilValue
  };
  for (size_t i = 0; i < u.size(); ++i) REAL(values[1])[i] = u[i];
  for (size_t i = 0; i < v.size(); ++i) REAL(values[2])[i] = v[i];
  SEXP out = namedList(names, values);
  UNPROTECT(1 + wantU + wantV);
  return out;
}

template <typename T>
SEXP lsFitT(SEXP x, SEXP n, SEXP p, SEXP y, SEXP ny, SEXP tol, SEXP coef,
            SEXP resid, SEXP effects)
{
  int rows = asInteger(n), cols = asInteger(p), ycols = asInteger(ny);
  std::vector<T>
    fx = narrow<T>(x, "x"), fy = narrow<T>(y, "y"),
    fcoef = narrow<T>(coef, "coefficients"),
    fresid = narrow<T>(resid, "residuals"),
    feff = narrow<T>(effects, "effects");

  const char * names[] = { "qr", "n", "p", "ny", "tol", "coefficients",
                           "residuals", "effects", "rank", "pivot", "qraux",
                           "" };
  SEXP values[] = {
    R_NilValue, n, p, ny, tol, R_NilValue, R_NilValue, R_NilValue,
    PROTECT(allocVector(INTSXP, 1)),
    PROTECT(allocVector(INTSXP, cols)),
    PROTECT(allocVector(REALSXP, cols))
  };
  int * rank = INTEGER(values[8]), * pivot = INTEGER(values[9]);
  double * qraux = REAL(values[10]);
  *rank = 0;
  for (int i = 0; i < cols; ++i) { pivot[i] = i; qraux[i] = 0.0; }

  leastSquaresFit(fx.data(), rows, cols, fy.data(), ycols, asReal(tol),
           fcoef.data(), fresid.data(), feff.data(), rank, pivot, qraux);

  values[0] = PROTECT(doublesLike(x, fx.data()));
  values[5] = PROTECT(doublesLike(coef, fcoef.data()));
  values[6] = PROTECT(doublesLike(resid, fresid.data()));
  values[7] = PROTECT(doublesLike(effects, feff.data()));
  SEXP out = namedList(names, values);
  UNPROTECT(7);
  return out;
}

template <typename T>
SEXP distT(SEXP points, int n, int k, SEXP method, SEXP p)
{
  std::vector<T> pts = narrow<T>(points, "points"), d((size_t) n * n);
  distance(pts.data(), k * sizeof(T), n, pts.data(), k * sizeof(T),
           n, k, d.data(), n * sizeof(T), getDistEnum(string(method)),
           (T) asReal(p));
  return doubles(d.data(), d.size());
}

// merge and order as hclust() has them, from the sub/sup merge pairs
template <typename T>
SEXP clusteringResult(int n, const int * sub, const int * sup,
                      const T * heights)
{
  const char * names[] = { "merge", "order", "height", "" };
  SEXP values[] = {
    PROTECT(allocVector(INTSXP, 2 * (n - 1))),
    PROTECT(allocVector(INTSXP, n)),
    PROTECT(doubles(heights, n - 1))
  };
  formatClustering(n - 1, sub, sup, INTEGER(values[0]), INTEGER(values[1]));
  SEXP out = namedList(names, values);
  UNPROTECT(3);
  return out;
}

// beta is the flexible methods' parameter; their update
// (1 - lambda)/2 (d(k,i) + d(k,j)) + beta d(i,j) takes lambda = beta
template <typename T>
SEXP hclustT(SEXP distMat, int n, SEXP method, T beta)
{
  std::vector<T> d = narrow<T>(distMat, "distances"), heights(n - 1);
  std::vector<int> sub(n - 1), sup(n - 1);
  hcluster(d.data(), n * sizeof(T), n, sub.data(), sup.data(),
           heights.data(), getClusterEnum(string(method)), beta, beta);
  return clusteringResult(n, sub.data(), sup.data(), heights.data());
}

template <typename T>
SEXP distClustT(SEXP distMethod, SEXP clustMethod, SEXP points, int n,
                int dim, T beta)
{
  std::vector<T> pts = narrow<T>(points, "points"), heights(n - 1);
  std::vector<int> sub(n - 1), sup(n - 1);

  size_t pitch = 0;
  T * gpuDistances = NULL;
  distanceLeaveOnGpu(getDistEnum(string(distMethod)), T(2), pts.data(),
                     dim, n, &gpuDistances, &pitch);
  hclusterPreparedDistances(gpuDistances, pitch, n, sub.data(), sup.data(),
                            heights.data(), getClusterEnum(string(clustMethod)),
                            beta, beta);
  cudaFree(gpuDistances);
  return clusteringResult(n, sub.data(), sup.data(), heights.data());
}

} // namespace

extern "C" {

SEXP gpuPmcc(SEXP use, SEXP x, SEXP nx, SEXP y, SEXP ny, SEXP size,
             SEXP precision)
{
  requireDevice();
  UseObs obs = everything;
  switch (asInteger(use)) {
  case 0: obs = everything; break;
  case 1: obs = pairwiseComplete; break;
  default: error("gputools: unknown use method");
  }
  int na = asInteger(nx), nb = asInteger(ny), n = asInteger(size);
  return isDouble(precision) ? pmccT<double>(obs, x, na, y, nb, n)
                             : pmccT<float>(obs, x, na, y, nb, n);
}

SEXP gpuCorProduct(SEXP x, SEXP y, SEXP precision)
{
  requireDevice();
  return isDouble(precision) ? corProductT<double>(x, y)
                             : corProductT<float>(x, y);
}

SEXP gpuCorPairs(SEXP x, SEXP y, SEXP mode, SEXP threshold, SEXP k,
                 SEXP absolute, SEXP tile, SEXP limit, SEXP precision)
{
  requireDevice();
  int md = asInteger(mode), kk = asInteger(k), tl = asInteger(tile);
  double thr = asReal(threshold), lim = asReal(limit);
  bool abs = asLogical(absolute);
  return isDouble(precision)
    ? corPairsT<double>(x, y, md, thr, kk, abs, tl, lim)
    : corPairsT<float>(x, y, md, thr, kk, abs, tl, lim);
}

SEXP gpuAssoc(SEXP x, SEXP y, SEXP q, SEXP mode, SEXP threshold, SEXP k,
              SEXP tile, SEXP limit, SEXP precision)
{
  requireDevice();
  int md = asInteger(mode), kk = asInteger(k), tl = asInteger(tile);
  double thr = asReal(threshold), lim = asReal(limit);
  return isDouble(precision)
    ? assocT<double>(x, y, q, md, thr, kk, tl, lim)
    : assocT<float>(x, y, q, md, thr, kk, tl, lim);
}

SEXP gpuKmeans(SEXP x, SEXP centers, SEXP iterMax, SEXP precision)
{
  requireDevice();
  int maxiter = asInteger(iterMax);
  return isDouble(precision) ? kmeansT<double>(x, centers, maxiter)
                             : kmeansT<float>(x, centers, maxiter);
}

SEXP gpuRank(SEXP x, SEXP precision)
{
  requireDevice();
  return isDouble(precision) ? rankT<double>(x) : rankT<float>(x);
}

SEXP gpuSetDevice(SEXP device)
{
  requireDevice();
  setDevice(asInteger(device));
  return R_NilValue;
}

SEXP gpuGetDevice()
{
  requireDevice();
  int device = 0;
  getDevice(&device);
  return ScalarInteger(device);
}

SEXP gpuTtest(SEXP goodPairs, SEXP coeffs, SEXP precision)
{
  requireDevice();
  return isDouble(precision) ? ttestT<double>(goodPairs, coeffs)
                             : ttestT<float>(goodPairs, coeffs);
}

SEXP gpuKendall(SEXP x, SEXP nx, SEXP y, SEXP ny, SEXP size,
                SEXP precision)
{
  requireDevice();
  int na = asInteger(nx), nb = asInteger(ny), n = asInteger(size);
  return isDouble(precision)
    ? rankCorT<double>("gpuKendall", x, na, y, nb, n)
    : rankCorT<float>("gpuKendall", x, na, y, nb, n);
}

SEXP gpuSpearman(SEXP x, SEXP nx, SEXP y, SEXP ny, SEXP size,
                 SEXP precision)
{
  requireDevice();
  int na = asInteger(nx), nb = asInteger(ny), n = asInteger(size);
  return isDouble(precision)
    ? rankCorT<double>("gpuSpearmanPairwise", x, na, y, nb, n)
    : rankCorT<float>("gpuSpearmanPairwise", x, na, y, nb, n);
}

SEXP gpuGranger(SEXP x, SEXP rows, SEXP cols, SEXP lag, SEXP precision)
{
  requireDevice();
  int r = asInteger(rows), n = asInteger(cols), p = asInteger(lag);
  return isDouble(precision) ? grangerT<double>(x, r, n, p)
                             : grangerT<float>(x, r, n, p);
}

SEXP gpuGrangerXY(SEXP x, SEXP rows, SEXP colsx, SEXP y, SEXP colsy,
                  SEXP lag, SEXP precision)
{
  requireDevice();
  int r = asInteger(rows), nx = asInteger(colsx), ny = asInteger(colsy),
    p = asInteger(lag);
  return isDouble(precision) ? grangerXYT<double>(x, r, nx, y, ny, p)
                             : grangerXYT<float>(x, r, nx, y, ny, p);
}

SEXP gpuDist(SEXP points, SEXP numPoints, SEXP dim, SEXP method, SEXP p,
             SEXP precision)
{
  requireDevice();
  int n = asInteger(numPoints), k = asInteger(dim);
  return isDouble(precision) ? distT<double>(points, n, k, method, p)
                             : distT<float>(points, n, k, method, p);
}

SEXP gpuHclust(SEXP distMat, SEXP numPoints, SEXP method, SEXP beta,
               SEXP precision)
{
  requireDevice();
  int n = asInteger(numPoints);
  double b = asReal(beta);
  return isDouble(precision) ? hclustT<double>(distMat, n, method, b)
                             : hclustT<float>(distMat, n, method, (float) b);
}

SEXP gpuDistClust(SEXP distMethod, SEXP clustMethod, SEXP points,
                  SEXP numPoints, SEXP dim, SEXP beta, SEXP precision)
{
  requireDevice();
  int n = asInteger(numPoints), k = asInteger(dim);
  double b = asReal(beta);
  return isDouble(precision)
    ? distClustT<double>(distMethod, clustMethod, points, n, k, b)
    : distClustT<float>(distMethod, clustMethod, points, n, k, (float) b);
}

SEXP gpuQrDecomp(SEXP x, SEXP tol, SEXP precision)
{
  requireDevice();
  return isDouble(precision) ? qrDecompT<double>(x, tol)
                             : qrDecompT<float>(x, tol);
}

// Solves x b = y, as solve(x, y): x is n x p (n >= p), y is n x k; a square
// x by LU, an overdetermined one by least squares.
SEXP gpuSolve(SEXP x, SEXP y, SEXP precision)
{
  requireDevice();
  return isDouble(precision) ? solveT<double>(x, y) : solveT<float>(x, y);
}

SEXP gpuSvd(SEXP x, SEXP wantU, SEXP wantV, SEXP precision)
{
  requireDevice();
  bool u = asLogical(wantU), v = asLogical(wantV);
  return isDouble(precision) ? svdT<double>(x, u, v) : svdT<float>(x, u, v);
}

SEXP gpuMutualInfo(SEXP bins, SEXP splineOrder, SEXP nsamples,
                   SEXP x, SEXP nx, SEXP y, SEXP ny, SEXP precision)
{
  requireDevice();
  int b = asInteger(bins), o = asInteger(splineOrder), n = asInteger(nsamples),
    na = asInteger(nx), nb = asInteger(ny);
  return isDouble(precision) ? mutualInfoT<double>(b, o, n, x, na, y, nb)
                             : mutualInfoT<float>(b, o, n, x, na, y, nb);
}

// Least squares by blocked, pivoted Householder QR. coef, resid and
// effects come in initialized (zeros, zeros, y) and go back filled in,
// with the attributes they came with.
SEXP gpuLSFit(SEXP x, SEXP n, SEXP p, SEXP y, SEXP ny, SEXP tol,
              SEXP coef, SEXP resid, SEXP effects, SEXP precision)
{
  requireDevice();
  return isDouble(precision)
    ? lsFitT<double>(x, n, p, y, ny, tol, coef, resid, effects)
    : lsFitT<float>(x, n, p, y, ny, tol, coef, resid, effects);
}

SEXP gputoolsBuildInfo()
{
  const char * names[] = { "cuda", "message", "" };
  SEXP out = PROTECT(mkNamed(VECSXP, names));
  SET_VECTOR_ELT(out, 0, ScalarLogical(TRUE));
  char version[64];
  snprintf(version, sizeof version, "built with CUDA %d.%d",
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
  SET_VECTOR_ELT(out, 1, mkString(version));
  UNPROTECT(1);
  return out;
}

#define CALLDEF(name, n) {#name, (DL_FUNC) &name, n},

static const R_CallMethodDef callMethods[] = {
  GPUTOOLS_GPU_ROUTINES(CALLDEF)
  GPUTOOLS_OTHER_ROUTINES(CALLDEF)
  {NULL, NULL, 0}
};

void R_init_gputools(DllInfo * dll)
{
  R_registerRoutines(dll, NULL, callMethods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}

} // extern "C"
