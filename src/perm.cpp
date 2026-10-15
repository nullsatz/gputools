#include <cmath>
#include <vector>

#include "cuda_runtime.h"

#include "R.h"

#include "cudaUtils.h"
#include "gpublas.h"

#include "perm.h"

template <typename T>
void permuteCorrelations(int n, int p, const T * zx, int q, const T * zy,
                         int B, const int * perms, double eps, double * r0,
                         int * counts, double * maxima)
{
  int self = zy == NULL;
  if (self) zy = zx;
  size_t pq = (size_t) p * q;
  T * dZx = gpuAlloc<T>((size_t) n * p), * dZy = gpuAlloc<T>((size_t) n * q),
    * dYp = gpuAlloc<T>((size_t) n * q), * dR0 = gpuAlloc<T>(pq),
    * dRb = gpuAlloc<T>(pq);
  int * dPerms = gpuAlloc<int>((size_t) n * B), * dCounts = gpuAlloc<int>(pq);
  CUDART_CALL(cudaMemcpy(dZx, zx, (size_t) n * p * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dZy, zy, (size_t) n * q * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dPerms, perms, (size_t) n * B * sizeof(int),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemset(dCounts, 0, pq * sizeof(int)));

  blas::gemm('T', 'N', p, q, n, T(1), dZx, n, dZy, n, T(0), dR0, p);

  T teps = (T) eps;
  size_t nq = (size_t) n * q;
  dim3 threads(256), gridY((unsigned) ((nq + 255) / 256)),
    gridR((unsigned) ((pq + 255) / 256));
  for (int b = 0; b < B; ++b) {
    int * dPerm = dPerms + (size_t) b * n;
    void * permArgs[] = { &dZy, &n, &q, &dPerm, &dYp };
    cudaLaunch("gpuPermuteRows", permArgs, gridY, threads, precisionOf<T>());
    blas::gemm('T', 'N', p, q, n, T(1), dZx, n, dYp, n, T(0), dRb, p);
    void * countArgs[] = { &dRb, &dR0, &p, &q, &self, &teps, &dCounts };
    cudaLaunch("gpuCountExceed", countArgs, gridR, threads, precisionOf<T>());
    // cuBLAS's index of the largest |value|, 1-based
    int at = blas::iamax((int) pq, dRb, 1);
    T top = 0;
    CUDART_CALL(cudaMemcpy(&top, dRb + (at - 1), sizeof(T),
                           cudaMemcpyDeviceToHost));
    maxima[b] = std::fabs((double) top);
  }

  std::vector<T> hR(pq);
  CUDART_CALL(cudaMemcpy(hR.data(), dR0, pq * sizeof(T),
                         cudaMemcpyDeviceToHost));
  for (size_t e = 0; e < pq; ++e) r0[e] = hR[e];
  CUDART_CALL(cudaMemcpy(counts, dCounts, pq * sizeof(int),
                         cudaMemcpyDeviceToHost));

  cudaFree(dZx); cudaFree(dZy); cudaFree(dYp); cudaFree(dR0); cudaFree(dRb);
  cudaFree(dPerms); cudaFree(dCounts);
}

template void permuteCorrelations<float>(int, int, const float *, int,
  const float *, int, const int *, double, double *, int *, double *);
template void permuteCorrelations<double>(int, int, const double *, int,
  const double *, int, const int *, double, double *, int *, double *);
