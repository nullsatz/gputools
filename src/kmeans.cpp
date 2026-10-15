#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "cuda_runtime.h"

#include "R.h"

#include "cudaUtils.h"
#include "gpublas.h"

#include "kmeans.h"

// Lloyd's k-means as R's kmeans_Lloyd (stats/src/kmeans.c) runs it: assign
// every point to its nearest centre; stop if no assignment changed;
// otherwise move each centre to the mean of its points (an empty cluster's
// centre becomes NaN) and repeat, at most maxiter times. The distances and
// the cluster sums are matrix products on the GPU, a block of points at a
// time; the centres are averaged on the host in double.
template <typename T>
void lloyd(int n, int d, const T * x, int k, double * centers, int maxiter,
           int * cluster, int * sizes, int * iterations)
{
  // a block of points against every centre fits in about 256 MB
  int block = (int) std::max<size_t>(1, std::min<size_t>(n,
    ((size_t) 1 << 26) / k));

  T * dX = gpuAlloc<T>((size_t) n * d), * dC = gpuAlloc<T>((size_t) k * d),
    * dNorm = gpuAlloc<T>(k), * dG = gpuAlloc<T>((size_t) block * k),
    * dS = gpuAlloc<T>((size_t) k * d);
  int * dLabels = gpuAlloc<int>(n), * dChanged = gpuAlloc<int>(1);
  CUDART_CALL(cudaMemcpy(dX, x, (size_t) n * d * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemset(dLabels, 0xff, n * sizeof(int)));   // all -1

  std::vector<T> hC((size_t) k * d), hNorm(k), hS((size_t) k * d);
  std::vector<int> labels(n);
  std::vector<double> counts(k);
  int iter;
  for (iter = 0; iter < maxiter; ++iter) {
    for (int j = 0; j < k; ++j) {
      double norm = 0;
      for (int c = 0; c < d; ++c) {
        double v = centers[j + (size_t) c * k];
        norm += v * v;
        hC[j + (size_t) c * k] = (T) v;
      }
      hNorm[j] = (T) norm;
    }
    CUDART_CALL(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(T),
                           cudaMemcpyHostToDevice));
    CUDART_CALL(cudaMemcpy(dNorm, hNorm.data(), k * sizeof(T),
                           cudaMemcpyHostToDevice));

    // assign
    CUDART_CALL(cudaMemset(dChanged, 0, sizeof(int)));
    for (int r0 = 0; r0 < n; r0 += block) {
      int nr = std::min(block, n - r0);
      blas::gemm('N', 'T', nr, k, d, T(1), dX + r0, n, dC, k, T(0), dG, nr);
      int * dL = dLabels + r0;
      dim3 grid((nr + 255) / 256), threads(256);
      void * args[] = { &dG, &nr, &k, &dNorm, &dL, &dChanged };
      cudaLaunch("gpuAssignNearest", args, grid, threads, precisionOf<T>());
    }
    int changed = 0;
    CUDART_CALL(cudaMemcpy(&changed, dChanged, sizeof(int),
                           cudaMemcpyDeviceToHost));
    if (!changed) break;

    // update: the sums of each cluster's points, a' x, block by block
    CUDART_CALL(cudaMemset(dS, 0, (size_t) k * d * sizeof(T)));
    for (int r0 = 0; r0 < n; r0 += block) {
      int nr = std::min(block, n - r0);
      int * dL = dLabels + r0;
      size_t entries = (size_t) nr * k;
      dim3 grid((unsigned) ((entries + 255) / 256)), threads(256);
      void * args[] = { &dL, &nr, &k, &dG };
      cudaLaunch("gpuOneHot", args, grid, threads, precisionOf<T>());
      blas::gemm('T', 'N', k, d, nr, T(1), dG, nr, dX + r0, n, T(1), dS, k);
    }
    CUDART_CALL(cudaMemcpy(hS.data(), dS, hS.size() * sizeof(T),
                           cudaMemcpyDeviceToHost));
    CUDART_CALL(cudaMemcpy(labels.data(), dLabels, n * sizeof(int),
                           cudaMemcpyDeviceToHost));
    std::fill(counts.begin(), counts.end(), 0.0);
    for (int i = 0; i < n; ++i) if (labels[i] >= 0) counts[labels[i]] += 1;
    for (int j = 0; j < k; ++j)
      for (int c = 0; c < d; ++c)
        centers[j + (size_t) c * k] = hS[j + (size_t) c * k] / counts[j];
  }
  *iterations = iter + 1;

  CUDART_CALL(cudaMemcpy(labels.data(), dLabels, n * sizeof(int),
                         cudaMemcpyDeviceToHost));
  std::fill(sizes, sizes + k, 0);
  for (int i = 0; i < n; ++i) {
    cluster[i] = labels[i] + 1;
    if (labels[i] >= 0) sizes[labels[i]]++;
  }

  cudaFree(dX); cudaFree(dC); cudaFree(dNorm); cudaFree(dG); cudaFree(dS);
  cudaFree(dLabels); cudaFree(dChanged);
}

template void lloyd<float>(int, int, const float *, int, double *, int,
  int *, int *, int *);
template void lloyd<double>(int, int, const double *, int, double *, int,
  int *, int *, int *);
