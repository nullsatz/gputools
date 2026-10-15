#include "cuda_runtime.h"

#include "R.h"

#include "cudaUtils.h"
#include "gpublas.h"

#include "rank.h"

// Ranks each column of the n x p matrix x (column-major on the host) on
// the GPU, as rank(ties.method = "average", na.last = "keep") would, into
// ranks (n x p). NaN marks a missing value.
template <typename T>
void rankColumns(const T * x, int n, int p, T * ranks)
{
  int n2 = 1;
  while (n2 < n) n2 <<= 1;
  T * dX = gpuAlloc<T>((size_t) n * p), * dKeys = gpuAlloc<T>((size_t) n2 * p),
    * dRanks = gpuAlloc<T>((size_t) n * p);
  int * dRows = gpuAlloc<int>((size_t) n2 * p);
  CUDART_CALL(cudaMemcpy(dX, x, (size_t) n * p * sizeof(T),
                         cudaMemcpyHostToDevice));

  dim3 grid(p), block(n2 < 512 ? n2 : 512);
  void * args[] = { &dX, &n, &n2, &dKeys, &dRows, &dRanks };
  cudaLaunch("gpuRankColumns", args, grid, block, precisionOf<T>());

  CUDART_CALL(cudaMemcpy(ranks, dRanks, (size_t) n * p * sizeof(T),
                         cudaMemcpyDeviceToHost));
  cudaFree(dX);
  cudaFree(dKeys);
  cudaFree(dRanks);
  cudaFree(dRows);
}

template void rankColumns<float>(const float * x, int n, int p, float * ranks);
template void rankColumns<double>(const double * x, int n, int p,
                                  double * ranks);
