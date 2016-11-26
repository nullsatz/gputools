#include "nvrtc.h"
#include "cuda.h"

#include "R.h"
#include "Rcpp.h"

#include "cuseful.h"
#include "cudaUtils.h"

#define NUMTHREADS 16

// [[Rcpp::export]]
Rcpp::NumericMatrix
kendall(const Rcpp::NumericMatrix & x,
        const Rcpp::NumericMatrix & y,
        const Rcpp::NumericVector & precisionFlag)
{
  size_t 
    nx = x.ncol(), ny = y.ncol(), sampleSize = x.nrow(),
    outputLength = nx * ny,
    outputBytes = outputLength * sizeof(double),
    xBytes = nx * sampleSize * sizeof(double), 
    yBytes = ny * sampleSize * sizeof(double); 
  double
    * gpux, * gpuy; 
  double
    * gpuResults;
  dim3
    grid(nx, ny), block(NUMTHREADS, NUMTHREADS);

  cudaMalloc((void **) & gpux, xBytes);
  cudaMalloc((void **) & gpuy, yBytes);
  checkCudaError("input vector space allocation");

  cudaMemcpy(gpux, &x[0], xBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuy, &y[0], yBytes, cudaMemcpyHostToDevice);
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
  cudaLaunch("gpuKendall<double>", args, grid, block);

  cudaFree(gpux);
  cudaFree(gpuy);

  Rcpp::NumericMatrix results(nx, ny);
  cudaMemcpy(&results[0], gpuResults, outputBytes, cudaMemcpyDeviceToHost);
  cudaFree(gpuResults);
  checkCudaError("copying results from gpu and cleaning up");

  return results;
}
