#ifndef _GPUBLAS_H_
#define _GPUBLAS_H_

// cuBLAS v2 behind the call shapes of the legacy cuBLAS API the QR and
// least-squares code was written against: no handle argument, scalars by
// value, transpose/side/fill as characters. Every call checks its status
// and raises an R error on failure.

#include <cstddef>

#include "cuda_runtime.h"
#include "cublas_v2.h"

// the current device's handle, created on first use
cublasHandle_t gpuBlasHandle();
void gpuBlasCheck(cublasStatus_t status, const char * call);
void gpuCudaCheck(cudaError_t status, const char * call);

#define CUBLAS_CALL(x) gpuBlasCheck((x), #x)
#define CUDART_CALL(x) gpuCudaCheck((x), #x)

template <typename T>
T * gpuAlloc(size_t n)
{
  T * p = NULL;
  CUDART_CALL(cudaMalloc((void **) &p, n * sizeof(T)));
  return p;
}

namespace blas {

inline cublasOperation_t op(char t)
{
  return (t == 'T' || t == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
}

inline void setMatrix(int rows, int cols, int elemSize, const void * a,
                      int lda, void * b, int ldb)
{
  CUBLAS_CALL(cublasSetMatrix(rows, cols, elemSize, a, lda, b, ldb));
}

inline void getMatrix(int rows, int cols, int elemSize, const void * a,
                      int lda, void * b, int ldb)
{
  CUBLAS_CALL(cublasGetMatrix(rows, cols, elemSize, a, lda, b, ldb));
}

inline void setVector(int n, int elemSize, const void * x, int incx,
                      void * y, int incy)
{
  CUBLAS_CALL(cublasSetVector(n, elemSize, x, incx, y, incy));
}

inline void getVector(int n, int elemSize, const void * x, int incx,
                      void * y, int incy)
{
  CUBLAS_CALL(cublasGetVector(n, elemSize, x, incx, y, incy));
}

inline void copy(int n, const float * x, int incx, float * y, int incy)
{
  CUBLAS_CALL(cublasScopy(gpuBlasHandle(), n, x, incx, y, incy));
}

inline void swap(int n, float * x, int incx, float * y, int incy)
{
  CUBLAS_CALL(cublasSswap(gpuBlasHandle(), n, x, incx, y, incy));
}

inline void scal(int n, float alpha, float * x, int incx)
{
  CUBLAS_CALL(cublasSscal(gpuBlasHandle(), n, &alpha, x, incx));
}

inline void axpy(int n, float alpha, const float * x, int incx,
                 float * y, int incy)
{
  CUBLAS_CALL(cublasSaxpy(gpuBlasHandle(), n, &alpha, x, incx, y, incy));
}

inline float dot(int n, const float * x, int incx, const float * y, int incy)
{
  float result;
  CUBLAS_CALL(cublasSdot(gpuBlasHandle(), n, x, incx, y, incy, &result));
  return result;
}

inline float nrm2(int n, const float * x, int incx)
{
  float result;
  CUBLAS_CALL(cublasSnrm2(gpuBlasHandle(), n, x, incx, &result));
  return result;
}

// one-based, as in BLAS
inline int iamax(int n, const float * x, int incx)
{
  int result;
  CUBLAS_CALL(cublasIsamax(gpuBlasHandle(), n, x, incx, &result));
  return result;
}

inline void gemv(char trans, int m, int n, float alpha, const float * a,
                 int lda, const float * x, int incx, float beta, float * y,
                 int incy)
{
  CUBLAS_CALL(cublasSgemv(gpuBlasHandle(), op(trans), m, n, &alpha, a, lda,
                          x, incx, &beta, y, incy));
}

inline void ger(int m, int n, float alpha, const float * x, int incx,
                const float * y, int incy, float * a, int lda)
{
  CUBLAS_CALL(cublasSger(gpuBlasHandle(), m, n, &alpha, x, incx, y, incy,
                         a, lda));
}

inline void gemm(char transa, char transb, int m, int n, int k, float alpha,
                 const float * a, int lda, const float * b, int ldb,
                 float beta, float * c, int ldc)
{
  CUBLAS_CALL(cublasSgemm(gpuBlasHandle(), op(transa), op(transb), m, n, k,
                          &alpha, a, lda, b, ldb, &beta, c, ldc));
}

inline void syrk(char uplo, char trans, int n, int k, float alpha,
                 const float * a, int lda, float beta, float * c, int ldc)
{
  cublasFillMode_t fill =
    (uplo == 'U' || uplo == 'u') ? CUBLAS_FILL_MODE_UPPER
                                 : CUBLAS_FILL_MODE_LOWER;
  CUBLAS_CALL(cublasSsyrk(gpuBlasHandle(), fill, op(trans), n, k, &alpha,
                          a, lda, &beta, c, ldc));
}

// side 'L', upper triangular, no transpose, non-unit diagonal is all the
// package uses
inline void trsmLeftUpper(int m, int n, float alpha, const float * a,
                          int lda, float * b, int ldb)
{
  CUBLAS_CALL(cublasStrsm(gpuBlasHandle(), CUBLAS_SIDE_LEFT,
                          CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                          CUBLAS_DIAG_NON_UNIT, m, n, &alpha, a, lda, b, ldb));
}

} // namespace blas

#endif /* _GPUBLAS_H_ */
