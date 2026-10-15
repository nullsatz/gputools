#include <string>

#include "cuda_runtime.h"

#include "R.h"

#include "cuseful.h"
#include "cudaUtils.h"
#include "gpublas.h"

#include "kendall.h"

#define NUMTHREADS 16

// Runs a rank-correlation kernel (gpuKendall or gpuSpearmanPairwise) over
// every pair of a column of x (nx columns) with a column of y (ny), each of
// sampleSize rows, NaN marking a missing value. results and counts are
// nx * ny, column-major: the correlation and the number of complete rows
// it used.
template <typename T>
void rankCorrelation(const char * kernel, const T * x, size_t nx,
                     const T * y, size_t ny, size_t sampleSize,
                     double * results, double * counts)
{
  size_t outputLength = nx * ny;
  T * gpux = gpuAlloc<T>(nx * sampleSize), * gpuy = gpuAlloc<T>(ny * sampleSize);
  double * gpuResults = gpuAlloc<double>(outputLength),
    * gpuCounts = gpuAlloc<double>(outputLength);
  CUDART_CALL(cudaMemcpy(gpux, x, nx * sampleSize * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(gpuy, y, ny * sampleSize * sizeof(T),
                         cudaMemcpyHostToDevice));

  dim3 grid(nx, ny), block(NUMTHREADS, NUMTHREADS);
  void * args[] = {
    &gpux, &nx, &gpuy, &ny, &sampleSize, &gpuResults, &gpuCounts
  };
  cudaLaunch(kernel, args, grid, block, precisionOf<T>());

  CUDART_CALL(cudaMemcpy(results, gpuResults, outputLength * sizeof(double),
                         cudaMemcpyDeviceToHost));
  CUDART_CALL(cudaMemcpy(counts, gpuCounts, outputLength * sizeof(double),
                         cudaMemcpyDeviceToHost));
  cudaFree(gpux);
  cudaFree(gpuy);
  cudaFree(gpuResults);
  cudaFree(gpuCounts);
}

template void rankCorrelation<float>(const char * kernel, const float * x,
  size_t nx, const float * y, size_t ny, size_t sampleSize, double * results,
  double * counts);
template void rankCorrelation<double>(const char * kernel, const double * x,
  size_t nx, const double * y, size_t ny, size_t sampleSize, double * results,
  double * counts);
