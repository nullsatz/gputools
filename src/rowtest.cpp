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

  // labellings a block at a time, so that genes x block fits in ~128 MB;
  // each block's product also carries the observed labelling as its
  // column 0, recomputed with the block (the same arithmetic, even where
  // the library picks kernels by width), so that a permutation equivalent
  // to the observed labelling always counts as reaching it
  int block = (int) std::max<size_t>(1, std::min<size_t>(nb,
    ((size_t) 1 << 25) / genes));
  T * dX = gpuAlloc<T>(gn), * dX2 = gpuAlloc<T>(gn), * dS = gpuAlloc<T>(genes),
    * dQ = gpuAlloc<T>(genes), * dG = gpuAlloc<T>((size_t) n * nb),
    * dGb = gpuAlloc<T>((size_t) n * (block + 1)),
    * dS1 = gpuAlloc<T>((size_t) genes * (block + 1)),
    * dQ1 = gpuAlloc<T>((size_t) genes * (block + 1)),
    * dT = gpuAlloc<T>((size_t) genes * (block + 1));
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
  std::vector<T> ht(genes);
  // the observed labelling (column 0 of labels) heads every block
  CUDART_CALL(cudaMemcpy(dGb, dG, n * sizeof(T), cudaMemcpyDeviceToDevice));
  for (int b0 = 1; b0 < std::max(nb, 2); b0 += block) {
    // permutations b0 .. b0 + c - 1 (none when nb == 1: the observed only)
    int c = std::min(block, nb - b0);
    if (c > 0)
      CUDART_CALL(cudaMemcpy(dGb + n, dG + (size_t) b0 * n,
                             (size_t) c * n * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    int cols = 1 + std::max(c, 0);
    blas::gemm('N', 'N', genes, cols, n, T(1), dX, genes, dGb, n, T(0), dS1,
               genes);
    blas::gemm('N', 'N', genes, cols, n, T(1), dX2, genes, dGb, n, T(0), dQ1,
               genes);
    size_t entries = (size_t) genes * cols;
    dim3 grid((unsigned) ((entries + 255) / 256)), threads(256);
    void * tArgs[] = { &dS1, &dQ1, &dS, &dQ, &genes, &cols, &n1, &n2, &pool,
                       &dT };
    cudaLaunch("gpuRowT", tArgs, grid, threads, precisionOf<T>());
    if (b0 == 1)
      CUDART_CALL(cudaMemcpy(ht.data(), dT, genes * sizeof(T),
                             cudaMemcpyDeviceToHost));
    if (c > 0) {
      int first = b0 - 1;
      dim3 cgrid(c);
      void * cArgs[] = { &dT, &genes, &first, &teps, &dCounts, &dMax };
      cudaLaunch("gpuRowTCount", cArgs, cgrid, threads, precisionOf<T>());
    }
    if (nb <= 1) break;
  }

  for (int r = 0; r < genes; ++r) t0[r] = ht[r];
  CUDART_CALL(cudaMemcpy(counts, dCounts, genes * sizeof(int),
                         cudaMemcpyDeviceToHost));
  if (nb > 1)
    CUDART_CALL(cudaMemcpy(maxima, dMax, (nb - 1) * sizeof(double),
                           cudaMemcpyDeviceToHost));

  cudaFree(dX); cudaFree(dX2); cudaFree(dS); cudaFree(dQ); cudaFree(dG);
  cudaFree(dGb); cudaFree(dS1); cudaFree(dQ1); cudaFree(dT);
  cudaFree(dCounts); cudaFree(dMax);
}

template void rowTtests<float>(int, int, const float *, int, const int *,
  int, bool, double, double *, int *, double *);
template void rowTtests<double>(int, int, const double *, int, const int *,
  int, bool, double, double *, int *, double *);
