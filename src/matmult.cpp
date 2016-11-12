#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<cuda_runtime.h>
#include<cublas_v2.h>

#include<cuseful.h>

#include<R.h>
#include<Rinternals.h>
#include<R_ext/BLAS.h>

#include<matmult.h>

SEXP gpuMatMult(SEXP a, SEXP b) {
  double
    * xa = REAL(a), * xb = REAL(b),
    * gpua, * gpub, * gpuc;

  SEXP
    dima = getAttrib(a, R_DimSymbol),
    dimb = getAttrib(b, R_DimSymbol);

  int
    rowsa = INTEGER(dima)[0], colsa = INTEGER(dima)[1],
    rowsb = INTEGER(dimb)[0], colsb = INTEGER(dimb)[1];

  cublasStatus_t stat;
  cublasHandle_t handle;

  cudaError_t cudaStat;

  cudaStat = cudaMalloc((void**) &gpua, rowsa * colsa * sizeof(double));
  if (cudaStat != cudaSuccess) error("device memory allocation failed");

  cudaStat = cudaMalloc((void**) &gpub, rowsb * colsb * sizeof(double));
  if (cudaStat != cudaSuccess) error("device memory allocation failed");

  int
    rowsOpA = rowsa, colsOpA = colsa, colsOpB = colsb;

  cudaStat = cudaMalloc((void**) &gpuc, rowsOpA * colsOpB * sizeof(double));
  if (cudaStat != cudaSuccess) error("device memory allocation failed");

  stat = cublasCreate(&handle);
  if(stat != CUBLAS_STATUS_SUCCESS) error("CUBLAS initialization failed\n");

  stat = cublasSetMatrix(rowsa, colsa, sizeof(double), xa, rowsa,
      gpua, rowsa);
  if(stat != CUBLAS_STATUS_SUCCESS) {
    cudaFree(gpuc);
    cudaFree(gpub);
    cudaFree(gpua);
    cublasDestroy(handle);
    error("data download failed\n");
  }

  stat = cublasSetMatrix(rowsb, colsb, sizeof(double), xb, rowsb,
      gpub, rowsb);
  if(stat != CUBLAS_STATUS_SUCCESS) {
    cudaFree(gpuc);
    cudaFree(gpub);
    cudaFree(gpua);
    cublasDestroy(handle);
    error("data download failed\n");
  }

  const double alpha = 1.0, beta = 0.0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsOpA, colsOpB, colsOpA,
      &alpha, (const double *) gpua, rowsa, (const double *) gpub, rowsb,
      &beta, gpuc, rowsOpA);

  SEXP ab, dimab;
  PROTECT(ab = allocVector(REALSXP, rowsOpA * colsOpB));
  PROTECT(dimab = allocVector(INTSXP, 2));
  INTEGER(dimab)[0] = rowsOpA; INTEGER(dimab)[1] = colsOpB;
  setAttrib(ab, R_DimSymbol, dimab);

  double * xab = REAL(ab);
  stat = cublasGetMatrix(rowsOpA, colsOpB, sizeof(double), gpuc, rowsOpA,
      xab, rowsOpA);
  if(stat != CUBLAS_STATUS_SUCCESS) {
    cudaFree(gpuc);
    cudaFree(gpub);
    cudaFree(gpua);
    cublasDestroy(handle);
    error("data upload failed\n");
  }

  cudaFree(gpua);
  cudaFree(gpub);
  cudaFree(gpuc);

  cublasDestroy(handle);
  UNPROTECT(2);
  return ab;
}
