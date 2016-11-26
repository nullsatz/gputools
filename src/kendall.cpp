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
//    xBytes = nx * sampleSize * sizeof(float), 
//    yBytes = ny * sampleSize * sizeof(float); 
  // float
  double
    * gpux, * gpuy; 
  double
    * gpuResults;
  dim3
    grid(nx, ny), block(NUMTHREADS, NUMTHREADS);

  Rprintf("== 1\n");

  cudaMalloc((void **) & gpux, xBytes);
  cudaMalloc((void **) & gpuy, yBytes);
  checkCudaError("input vector space allocation");

  Rprintf("== 2\n");

  cudaMemcpy(gpux, &x[0], xBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuy, &y[0], yBytes, cudaMemcpyHostToDevice);
  checkCudaError("copying input vectors to gpu");

  Rprintf("== 3\n");

  cudaMalloc((void **) & gpuResults, outputBytes);
  checkCudaError("allocation of space for result matrix");

  Rprintf("== 4\n");

  void *args[] =
    { &gpux
    , &nx
    , &gpuy
    , &ny
    , &sampleSize
    , &gpuResults
    };
  cudaLaunch("gpuKendall<double>", args, grid, block);
//  cudaLaunch("gpuKendall<float>", args, grid, block);
//
  Rprintf("== 5\n");

  cudaFree(gpux);
  cudaFree(gpuy);

  Rprintf("== 6\n");

  Rcpp::NumericMatrix results(nx, ny);
  cudaMemcpy(&results[0], gpuResults, outputBytes, cudaMemcpyDeviceToHost);
  Rprintf("== 7\n");
  cudaFree(gpuResults);
  checkCudaError("copying results from gpu and cleaning up");
  Rprintf("== 8\n");

  return results;
}
