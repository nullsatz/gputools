#include <algorithm>
#include <vector>

#include "cuda_runtime.h"

#include "R.h"

#include "cudaUtils.h"
#include "gpublas.h"

#include "rowtest.h"

template <typename T>
void rowTtests(int genes, int n, const T * x, int nb, const int * labels,
               int n1, bool pooled, double eps, double * t0, int * counts,
               double * maxima)
{
  size_t gn = (size_t) genes * n;
  std::vector<T> x2(gn), s(genes, 0), q(genes, 0), g((size_t) n * nb);
  for (size_t e = 0; e < gn; ++e) x2[e] = x[e] * x[e];
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < genes; ++r) {
      s[r] += x[r + (size_t) c * genes];
      q[r] += x2[r + (size_t) c * genes];
    }
  for (size_t e = 0; e < g.size(); ++e) g[e] = (T) labels[e];

  // labellings a block at a time, so that genes x block fits in ~128 MB
  int block = (int) std::max<size_t>(1, std::min<size_t>(nb,
    ((size_t) 1 << 25) / genes));
  T * dX = gpuAlloc<T>(gn), * dX2 = gpuAlloc<T>(gn), * dS = gpuAlloc<T>(genes),
    * dQ = gpuAlloc<T>(genes), * dG = gpuAlloc<T>((size_t) n * nb),
    * dS1 = gpuAlloc<T>((size_t) genes * block),
    * dQ1 = gpuAlloc<T>((size_t) genes * block),
    * dT = gpuAlloc<T>((size_t) genes * block), * dT0 = gpuAlloc<T>(genes);
  int * dCounts = gpuAlloc<int>(genes);
  double * dMax = gpuAlloc<double>(nb > 1 ? nb - 1 : 1);
  CUDART_CALL(cudaMemcpy(dX, x, gn * sizeof(T), cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dX2, x2.data(), gn * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dS, s.data(), genes * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dQ, q.data(), genes * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dG, g.data(), g.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemset(dCounts, 0, genes * sizeof(int)));

  int n2 = n - n1, pool = pooled;
  T teps = (T) eps;
  for (int b0 = 0; b0 < nb; b0 += block) {
    int c = std::min(block, nb - b0);
    T * dGc = dG + (size_t) b0 * n;
    blas::gemm('N', 'N', genes, c, n, T(1), dX, genes, dGc, n, T(0), dS1,
               genes);
    blas::gemm('N', 'N', genes, c, n, T(1), dX2, genes, dGc, n, T(0), dQ1,
               genes);
    size_t entries = (size_t) genes * c;
    dim3 grid((unsigned) ((entries + 255) / 256)), threads(256);
    void * tArgs[] = { &dS1, &dQ1, &dS, &dQ, &genes, &c, &n1, &n2, &pool,
                       &dT };
    cudaLaunch("gpuRowT", tArgs, grid, threads, precisionOf<T>());
    if (b0 == 0)
      CUDART_CALL(cudaMemcpy(dT0, dT, genes * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    if (nb > 1) {
      dim3 cgrid(c);
      void * cArgs[] = { &dT, &dT0, &genes, &b0, &teps, &dCounts, &dMax };
      cudaLaunch("gpuRowTCount", cArgs, cgrid, threads, precisionOf<T>());
    }
  }

  std::vector<T> ht(genes);
  CUDART_CALL(cudaMemcpy(ht.data(), dT0, genes * sizeof(T),
                         cudaMemcpyDeviceToHost));
  for (int r = 0; r < genes; ++r) t0[r] = ht[r];
  CUDART_CALL(cudaMemcpy(counts, dCounts, genes * sizeof(int),
                         cudaMemcpyDeviceToHost));
  if (nb > 1)
    CUDART_CALL(cudaMemcpy(maxima, dMax, (nb - 1) * sizeof(double),
                           cudaMemcpyDeviceToHost));

  cudaFree(dX); cudaFree(dX2); cudaFree(dS); cudaFree(dQ); cudaFree(dG);
  cudaFree(dS1); cudaFree(dQ1); cudaFree(dT); cudaFree(dT0);
  cudaFree(dCounts); cudaFree(dMax);
}

template void rowTtests<float>(int, int, const float *, int, const int *,
  int, bool, double, double *, int *, double *);
template void rowTtests<double>(int, int, const double *, int, const int *,
  int, bool, double, double *, int *, double *);
