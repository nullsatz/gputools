#ifndef _GPUSOLVER_H_
#define _GPUSOLVER_H_

// cuSOLVER dense routines, overloaded on float/double like gpublas.h, with
// the workspace handled inside. Every call checks its status and raises an
// R error on failure; the LAPACK-style info results are returned for the
// caller to interpret.

#include "cusolverDn.h"

#include "gpublas.h"

// the current device's handle, created on first use
cusolverDnHandle_t gpuSolverHandle();
void gpuSolverCheck(cusolverStatus_t status, const char * call);

#define CUSOLVER_CALL(x) gpuSolverCheck((x), #x)

namespace solver {

// LU with partial pivoting of the n x n matrix a, in place. Returns info:
// 0, or i > 0 if U[i, i] is exactly zero.
#define GPUTOOLS_GETRF(T, P)                                                \
inline int getrf(int n, T * a, int lda, int * dPivots)                      \
{                                                                           \
  int lwork = 0;                                                            \
  CUSOLVER_CALL(cusolverDn##P##getrf_bufferSize(gpuSolverHandle(), n, n,    \
                                                a, lda, &lwork));           \
  T * work = gpuAlloc<T>(lwork > 0 ? lwork : 1);                            \
  int * dInfo = gpuAlloc<int>(1), info = 0;                                 \
  CUSOLVER_CALL(cusolverDn##P##getrf(gpuSolverHandle(), n, n, a, lda, work, \
                                     dPivots, dInfo));                      \
  CUDART_CALL(cudaMemcpy(&info, dInfo, sizeof(int),                         \
                         cudaMemcpyDeviceToHost));                          \
  cudaFree(work);                                                           \
  cudaFree(dInfo);                                                          \
  return info;                                                              \
}
GPUTOOLS_GETRF(float, S)
GPUTOOLS_GETRF(double, D)
#undef GPUTOOLS_GETRF

// solves A X = B for the nrhs columns of b, given getrf's factors of A
#define GPUTOOLS_GETRS(T, P)                                                \
inline void getrs(int n, int nrhs, const T * a, int lda,                    \
                  const int * dPivots, T * b, int ldb)                      \
{                                                                           \
  int * dInfo = gpuAlloc<int>(1);                                           \
  CUSOLVER_CALL(cusolverDn##P##getrs(gpuSolverHandle(), CUBLAS_OP_N, n,     \
                                     nrhs, a, lda, dPivots, b, ldb, dInfo));\
  cudaFree(dInfo);                                                          \
}
GPUTOOLS_GETRS(float, S)
GPUTOOLS_GETRS(double, D)
#undef GPUTOOLS_GETRS

// Householder QR of the m x n matrix a (m >= n), in place, LAPACK's form:
// R on and above the diagonal, the reflectors below it with scalars tau
#define GPUTOOLS_GEQRF(T, P)                                                \
inline void geqrf(int m, int n, T * a, int lda, T * tau)                    \
{                                                                           \
  int lwork = 0;                                                            \
  CUSOLVER_CALL(cusolverDn##P##geqrf_bufferSize(gpuSolverHandle(), m, n,    \
                                                a, lda, &lwork));           \
  T * work = gpuAlloc<T>(lwork > 0 ? lwork : 1);                            \
  int * dInfo = gpuAlloc<int>(1);                                           \
  CUSOLVER_CALL(cusolverDn##P##geqrf(gpuSolverHandle(), m, n, a, lda, tau,  \
                                     work, lwork, dInfo));                  \
  cudaFree(work);                                                           \
  cudaFree(dInfo);                                                          \
}
GPUTOOLS_GEQRF(float, S)
GPUTOOLS_GEQRF(double, D)
#undef GPUTOOLS_GEQRF

// c = Q^T c for the m x nrhs matrix c, Q from geqrf's k reflectors
#define GPUTOOLS_ORMQR(T, P)                                                \
inline void ormqrTransposeLeft(int m, int nrhs, int k, const T * a, int lda,\
                               const T * tau, T * c, int ldc)               \
{                                                                           \
  int lwork = 0;                                                            \
  CUSOLVER_CALL(cusolverDn##P##ormqr_bufferSize(gpuSolverHandle(),          \
    CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, k, a, lda, tau, c, ldc,         \
    &lwork));                                                               \
  T * work = gpuAlloc<T>(lwork > 0 ? lwork : 1);                            \
  int * dInfo = gpuAlloc<int>(1);                                           \
  CUSOLVER_CALL(cusolverDn##P##ormqr(gpuSolverHandle(), CUBLAS_SIDE_LEFT,   \
    CUBLAS_OP_T, m, nrhs, k, a, lda, tau, c, ldc, work, lwork, dInfo));     \
  cudaFree(work);                                                           \
  cudaFree(dInfo);                                                          \
}
GPUTOOLS_ORMQR(float, S)
GPUTOOLS_ORMQR(double, D)
#undef GPUTOOLS_ORMQR

} // namespace solver

#endif /* _GPUSOLVER_H_ */
