#include <string>
#include <algorithm>

#include "nvrtc.h"
#include "cuda.h"

#include "R.h"
#include "Rcpp.h"

#include "cuseful.h"
#include "cudaUtils.h"

#define NUMTHREADS 16

template <typename T>
Rcpp::NumericMatrix kendallHelper(const T * x, size_t nx,
                                  const T * y, size_t ny,
                                  size_t sampleSize,
                                  std::string kernel)
{
  size_t
    outputLength = nx * ny,
    outputBytes = outputLength * sizeof(double);
  double * gpuResults;

  dim3
    grid(nx, ny),
    block(NUMTHREADS, NUMTHREADS);

  size_t
    xBytes = nx * sampleSize * sizeof(T),
    yBytes = ny * sampleSize * sizeof(T);
  T * gpux, * gpuy;

  cudaMalloc((void **) & gpux, xBytes);
  cudaMalloc((void **) & gpuy, yBytes);
  checkCudaError("input vector space allocation");

  cudaMemcpy(gpux, x, xBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuy, y, yBytes, cudaMemcpyHostToDevice);
  checkCudaError("copying input vectors to gpu");

  cudaMalloc((void **) & gpuResults, outputBytes);
  checkCudaError("allocation of space for result matrix");

  void *args[] =
    { &gpux
    , &nx
    , &gpuy
    , &ny
    , &sampleSize
    , &gpuResults
    };
  cudaLaunch(kernel, args, grid, block);

  cudaFree(gpux);
  cudaFree(gpuy);

  Rcpp::NumericMatrix results(nx, ny);
  cudaMemcpy(&results[0], gpuResults, outputBytes, cudaMemcpyDeviceToHost);
  cudaFree(gpuResults);
  checkCudaError("copying results from gpu and cleaning up");

  return results;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix
kendall(const Rcpp::NumericMatrix & x,
        const Rcpp::NumericMatrix & y,
        const Rcpp::NumericVector & precisionFlag)
{
  int prec = (int) precisionFlag[0];

  size_t 
    nx = x.ncol(), ny = y.ncol(),
    sampleSize = x.nrow();

  Rcpp::NumericMatrix results;

  if (prec == 1) {
    float
      * xf = Calloc(x.length(), float),
      * yf = Calloc(y.length(), float);

    for(int i = 0; i < x.length(); ++i) {
      xf[i] = x[i];
    }
    for(int i = 0; i < y.length(); ++i) {
      yf[i] = y[i];
    }

    results = kendallHelper(xf, nx, yf, ny, sampleSize,
                            "gpuKendall<float>");
    Free(xf);
    Free(yf);
  } else if (prec == 2) {
    results = kendallHelper(&x[0], nx, &y[0], ny, sampleSize,
                            "gpuKendall<double>");
  } else {
    error("precision must be 1 (float) or 2 (double)");
  }
  return results;
}
