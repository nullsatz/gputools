#include <stdio.h>
#include <string.h>
#include <vector>

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

#include "rinterface.h"

// The .Call entry points. The kernels work in single precision: inputs
// arrive as R doubles and are narrowed here, results are widened back to
// doubles, so R never sees a "Csingle" object.

namespace {

// Narrows a numeric vector to floats. Unless naok, refuses NA, NaN and Inf
// the way .C() used to.
std::vector<float> floats(SEXP x, const char * what, bool naok = false)
{
  if (!isNumeric(x) && !isLogical(x))
    error("gputools: %s must be numeric", what);
  x = PROTECT(coerceVector(x, REALSXP));
  R_xlen_t n = XLENGTH(x);
  const double * px = REAL(x);
  std::vector<float> out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    if (!naok && !R_FINITE(px[i]))
      error("gputools: NA/NaN/Inf in %s", what);
    out[i] = (float) px[i];
  }
  UNPROTECT(1);
  return out;
}

// A new double vector holding v.
SEXP doubles(const float * v, R_xlen_t n)
{
  SEXP out = allocVector(REALSXP, n);
  double * po = REAL(out);
  for (R_xlen_t i = 0; i < n; ++i) po[i] = v[i];
  return out;
}

// A double copy of like (dims and all) overwritten with v.
SEXP doublesLike(SEXP like, const float * v)
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

} // namespace

dist_method getDistEnum(const char * methodStr)
{
  if(0 == strcmp(methodStr,"maximum"))    return MAXIMUM;
  if(0 == strcmp(methodStr,"manhattan"))  return MANHATTAN;
  if(0 == strcmp(methodStr,"canberra"))   return CANBERRA;
  if(0 == strcmp(methodStr,"binary"))     return BINARY;
  if(0 == strcmp(methodStr,"minkowski"))  return MINKOWSKI;
  //      if(0 == strcmp(methodStr,"dot"))                return DOT;
  return EUCLIDEAN;
}

hc_method getClusterEnum(const char * methodStr)
{
  if(0 == strcmp(methodStr,"complete"))           return COMPLETE;
  if(0 == strcmp(methodStr,"wpgma"))                      return WPGMA;
  if(0 == strcmp(methodStr,"average"))            return AVERAGE;
  if(0 == strcmp(methodStr,"median"))                     return MEDIAN;
  if(0 == strcmp(methodStr,"centroid"))           return CENTROID;
  if(0 == strcmp(methodStr,"flexible_group"))     return FLEXIBLE_GROUP;
  if(0 == strcmp(methodStr,"flexible"))           return FLEXIBLE;
  if(0 == strcmp(methodStr,"ward"))                       return WARD;
  if(0 == strcmp(methodStr,"mcquitty"))           return MCQUITTY;
  return SINGLE;
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

extern "C" {

SEXP gpuPmcc(SEXP use, SEXP x, SEXP nx, SEXP y, SEXP ny, SEXP size)
{
  UseObs obs;
  switch (asInteger(use)) {
  case 0: obs = everything; break;
  case 1: obs = pairwiseComplete; break;
  default: error("gputools: unknown use method");
  }
  int na = asInteger(nx), nb = asInteger(ny);
  std::vector<float>
    a = floats(x, "x", true), b = floats(y, "y", true),
    pairs(na * nb), corr(na * nb), ts(na * nb);
  pmcc(obs, a.data(), na, b.data(), nb, asInteger(size),
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

SEXP gpuSetDevice(SEXP device)
{
  setDevice(asInteger(device));
  return R_NilValue;
}

SEXP gpuGetDevice()
{
  int device = 0;
  getDevice(&device);
  return ScalarInteger(device);
}

SEXP gpuTtest(SEXP goodPairs, SEXP coeffs)
{
  std::vector<float>
    pairs = floats(goodPairs, "goodPairs", true),
    cors = floats(coeffs, "coeffs", true),
    ts(pairs.size());
  if (cors.size() != pairs.size())
    error("gputools: goodPairs and coeffs differ in length");
  testSignif(pairs.data(), cors.data(), pairs.size(), ts.data());
  return doubles(ts.data(), ts.size());
}

SEXP gpuKendall(SEXP x, SEXP nx, SEXP y, SEXP ny, SEXP size)
{
  int na = asInteger(nx), nb = asInteger(ny);
  std::vector<float> a = floats(x, "x"), b = floats(y, "y");
  SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t) na * nb));
  masterKendall(a.data(), na, b.data(), nb, asInteger(size), REAL(out));
  UNPROTECT(1);
  return out;
}

SEXP gpuGranger(SEXP x, SEXP rows, SEXP cols, SEXP lag)
{
  int n = asInteger(cols);
  std::vector<float> a = floats(x, "x"), f(n * n), p(n * n);
  granger(asInteger(rows), n, a.data(), asInteger(lag), f.data(), p.data());

  const char * names[] = { "fStats", "pValues", "" };
  SEXP values[] = {
    PROTECT(doubles(f.data(), n * n)), PROTECT(doubles(p.data(), n * n))
  };
  SEXP out = namedList(names, values);
  UNPROTECT(2);
  return out;
}

SEXP gpuGrangerXY(SEXP x, SEXP rows, SEXP colsx, SEXP y, SEXP colsy,
                  SEXP lag)
{
  int nx = asInteger(colsx), ny = asInteger(colsy);
  std::vector<float>
    a = floats(x, "x"), b = floats(y, "y"), f(nx * ny), p(nx * ny);
  grangerxy(asInteger(rows), nx, a.data(), ny, b.data(), asInteger(lag),
            f.data(), p.data());

  const char * names[] = { "fStats", "pValues", "" };
  SEXP values[] = {
    PROTECT(doubles(f.data(), nx * ny)), PROTECT(doubles(p.data(), nx * ny))
  };
  SEXP out = namedList(names, values);
  UNPROTECT(2);
  return out;
}

SEXP gpuDist(SEXP points, SEXP numPoints, SEXP dim, SEXP method, SEXP p)
{
  int n = asInteger(numPoints), k = asInteger(dim);
  std::vector<float> pts = floats(points, "points"), d((size_t) n * n);
  distance(pts.data(), k * sizeof(float), n, pts.data(), k * sizeof(float),
           n, k, d.data(), n * sizeof(float), getDistEnum(string(method)),
           (float) asReal(p));
  return doubles(d.data(), d.size());
}

// merge and order as hclust() has them, from the sub/sup merge pairs
SEXP clusteringResult(int n, const int * sub, const int * sup,
                      const float * heights)
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

SEXP gpuHclust(SEXP distMat, SEXP numPoints, SEXP method)
{
  int n = asInteger(numPoints);
  std::vector<float> d = floats(distMat, "distances"), heights(n - 1);
  std::vector<int> sub(n - 1), sup(n - 1);
  hcluster(d.data(), n * sizeof(float), n, sub.data(), sup.data(),
           heights.data(), getClusterEnum(string(method)), 0.5f, 0.5f);
  return clusteringResult(n, sub.data(), sup.data(), heights.data());
}

SEXP gpuDistClust(SEXP distMethod, SEXP clustMethod, SEXP points,
                  SEXP numPoints, SEXP dim)
{
  int n = asInteger(numPoints);
  std::vector<float> pts = floats(points, "points"), heights(n - 1);
  std::vector<int> sub(n - 1), sup(n - 1);

  size_t pitch = 0;
  float * gpuDistances = NULL;
  distanceLeaveOnGpu(getDistEnum(string(distMethod)), 2.f, pts.data(),
                     asInteger(dim), n, &gpuDistances, &pitch);
  hclusterPreparedDistances(gpuDistances, pitch, n, sub.data(), sup.data(),
                            heights.data(), getClusterEnum(string(clustMethod)),
                            0.5f, 0.5f);
  return clusteringResult(n, sub.data(), sup.data(), heights.data());
}

SEXP gpuQrDecomp(SEXP x, SEXP tol)
{
  SEXP dim = getAttrib(x, R_DimSymbol);
  int rows = INTEGER(dim)[0], cols = INTEGER(dim)[1];
  std::vector<float> a = floats(x, "x");

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

  float * dQR;
  cudaMalloc((void **) &dQR, a.size() * sizeof(float));
  checkCudaError("gpuQr: allocating the matrix");
  cudaMemcpy(dQR, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
  getQRDecompRR(rows, cols, asReal(tol), dQR, pivot, REAL(values[2]),
                INTEGER(values[3]));
  cudaMemcpy(a.data(), dQR, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dQR);
  checkCudaError("gpuQr: copying back the decomposition");

  values[0] = PROTECT(doublesLike(x, a.data()));
  SEXP out = namedList(names, values);
  UNPROTECT(4);
  return out;
}

SEXP gpuInverseFromQR(SEXP q, SEXP r, SEXP rows, SEXP cols)
{
  int n = asInteger(rows), p = asInteger(cols);
  std::vector<float> fq = floats(q, "q"), fr = floats(r, "r"), inv(n * p);

  float * dQ, * dR, * dInverse;
  cudaMalloc((void **) &dQ, n * p * sizeof(float));
  cudaMalloc((void **) &dR, p * p * sizeof(float));
  cudaMalloc((void **) &dInverse, n * p * sizeof(float));
  checkCudaError("gpuSolve: allocating");
  cudaMemcpy(dQ, fq.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dR, fr.data(), p * p * sizeof(float), cudaMemcpyHostToDevice);

  getInverseFromQR(n, p, dQ, dR, dInverse);

  cudaMemcpy(inv.data(), dInverse, n * p * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(dQ);
  cudaFree(dR);
  cudaFree(dInverse);
  checkCudaError("gpuSolve: copying back the inverse");
  return doubles(inv.data(), inv.size());
}

SEXP gpuSolveFromQR(SEXP q, SEXP r, SEXP y, SEXP rows, SEXP cols)
{
  int n = asInteger(rows), p = asInteger(cols);
  std::vector<float>
    fq = floats(q, "q"), fr = floats(r, "r"), fy = floats(y, "y"), b(p);
  solveFromQR(n, p, fq.data(), fr.data(), fy.data(), b.data());
  return doubles(b.data(), p);
}

SEXP gpuMutualInfo(SEXP bins, SEXP splineOrder, SEXP nsamples,
                   SEXP x, SEXP nx, SEXP y, SEXP ny)
{
  int na = asInteger(nx), nb = asInteger(ny);
  std::vector<float> a = floats(x, "x"), b = floats(y, "y"), mi(na * nb);
  bSplineMutualInfo(asInteger(bins), asInteger(splineOrder),
                    asInteger(nsamples), na, a.data(), nb, b.data(),
                    mi.data());
  return doubles(mi.data(), mi.size());
}

// Least squares by blocked, pivoted Householder QR. coef, resid and
// effects come in initialized (zeros, zeros, y) and go back filled in,
// with the attributes they came with.
SEXP gpuLSFit(SEXP x, SEXP n, SEXP p, SEXP y, SEXP ny, SEXP tol,
              SEXP coef, SEXP resid, SEXP effects)
{
  int rows = asInteger(n), cols = asInteger(p), ycols = asInteger(ny);
  std::vector<float>
    fx = floats(x, "x"), fy = floats(y, "y"),
    fcoef = floats(coef, "coefficients"), fresid = floats(resid, "residuals"),
    feff = floats(effects, "effects");

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

  gpuLSFitF(fx.data(), rows, cols, fy.data(), ycols, asReal(tol),
            fcoef.data(), fresid.data(), feff.data(), rank, pivot, qraux);

  values[0] = PROTECT(doublesLike(x, fx.data()));
  values[5] = PROTECT(doublesLike(coef, fcoef.data()));
  values[6] = PROTECT(doublesLike(resid, fresid.data()));
  values[7] = PROTECT(doublesLike(effects, feff.data()));
  SEXP out = namedList(names, values);
  UNPROTECT(7);
  return out;
}

#define CALLDEF(name, n) {#name, (DL_FUNC) &name, n}

static const R_CallMethodDef callMethods[] = {
  CALLDEF(gpuMatMult, 2),
  CALLDEF(gpuPmcc, 6),
  CALLDEF(gpuSetDevice, 1),
  CALLDEF(gpuGetDevice, 0),
  CALLDEF(gpuTtest, 2),
  CALLDEF(gpuKendall, 5),
  CALLDEF(gpuGranger, 4),
  CALLDEF(gpuGrangerXY, 6),
  CALLDEF(gpuDist, 5),
  CALLDEF(gpuHclust, 3),
  CALLDEF(gpuDistClust, 5),
  CALLDEF(gpuQrDecomp, 2),
  CALLDEF(gpuInverseFromQR, 4),
  CALLDEF(gpuSolveFromQR, 5),
  CALLDEF(gpuMutualInfo, 7),
  CALLDEF(gpuLSFit, 9),
  CALLDEF(gputoolsSetKernelDir, 1),
  CALLDEF(gputoolsUnloadKernels, 0),
  {NULL, NULL, 0}
};

void R_init_gputools(DllInfo * dll)
{
  R_registerRoutines(dll, NULL, callMethods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}

} // extern "C"
