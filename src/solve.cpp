#include <cmath>
#include <limits>
#include <vector>

#include "R.h"

#include "gpusolver.h"
#include "solve.h"

// Solves A X = B on the GPU for A (n x p, n >= p) and B (n x k), both
// column-major on the host, writing X (p x k). A square A is solved by LU
// with partial pivoting, as solve() does through LAPACK; an overdetermined
// one in the least-squares sense by Householder QR.
template <typename T>
void solveLinear(int n, int p, const T * a, int k, const T * b, T * x)
{
  T * dA = gpuAlloc<T>((size_t) n * p), * dB = gpuAlloc<T>((size_t) n * k);
  CUDART_CALL(cudaMemcpy(dA, a, (size_t) n * p * sizeof(T),
                         cudaMemcpyHostToDevice));
  CUDART_CALL(cudaMemcpy(dB, b, (size_t) n * k * sizeof(T),
                         cudaMemcpyHostToDevice));

  if (n == p) {
    int * dPivots = gpuAlloc<int>(n);
    int info = solver::getrf(n, dA, n, dPivots);
    if (info > 0) {
      cudaFree(dPivots); cudaFree(dA); cudaFree(dB);
      error("gputools: the system is exactly singular: U[%d,%d] = 0",
            info, info);
    }
    solver::getrs(n, k, dA, n, dPivots, dB, n);
    cudaFree(dPivots);
  } else {
    T * dTau = gpuAlloc<T>(p);
    solver::geqrf(n, p, dA, n, dTau);

    // R must be nonsingular for a unique least-squares solution
    std::vector<T> diag(p);
    blas::getVector(p, sizeof(T), dA, n + 1, diag.data(), 1);
    T largest = 0;
    for (T d : diag) largest = std::fmax(largest, std::fabs(d));
    for (int i = 0; i < p; ++i)
      if (!(std::fabs(diag[i]) > largest * 100 * p *
            std::numeric_limits<T>::epsilon())) {
        cudaFree(dTau); cudaFree(dA); cudaFree(dB);
        error("gputools: x is rank deficient (column %d), so the "
              "least-squares solution is not unique", i + 1);
      }

    solver::ormqrTransposeLeft(n, k, p, dA, n, dTau, dB, n);  // Q^T B
    blas::trsmLeftUpper(p, k, T(1), dA, n, dB, n);           // R X = (Q^T B)
    cudaFree(dTau);
  }

  // X is the first p rows of dB
  CUDART_CALL(cudaMemcpy2D(x, p * sizeof(T), dB, n * sizeof(T),
                           p * sizeof(T), k, cudaMemcpyDeviceToHost));
  cudaFree(dA);
  cudaFree(dB);
}

template void solveLinear<float>(int n, int p, const float * a, int k,
                                 const float * b, float * x);
template void solveLinear<double>(int n, int p, const double * a, int k,
                                  const double * b, double * x);
