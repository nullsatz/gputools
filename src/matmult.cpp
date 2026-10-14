#include "R.h"
#include "Rinternals.h"

#include "cudaUtils.h"
#include "gpublas.h"
#include "matmult.h"

SEXP gpuMatMult(SEXP a, SEXP b) {
  requireDevice();

  SEXP
    dima = getAttrib(a, R_DimSymbol),
    dimb = getAttrib(b, R_DimSymbol);

  int
    rowsa = INTEGER(dima)[0], colsa = INTEGER(dima)[1],
    rowsb = INTEGER(dimb)[0], colsb = INTEGER(dimb)[1];

  SEXP ab = PROTECT(allocMatrix(REALSXP, rowsa, colsb));

  double
    * gpua = gpuAlloc<double>((size_t) rowsa * colsa),
    * gpub = gpuAlloc<double>((size_t) rowsb * colsb),
    * gpuc = gpuAlloc<double>((size_t) rowsa * colsb);

  blas::setMatrix(rowsa, colsa, sizeof(double), REAL(a), rowsa, gpua, rowsa);
  blas::setMatrix(rowsb, colsb, sizeof(double), REAL(b), rowsb, gpub, rowsb);

  const double alpha = 1.0, beta = 0.0;
  CUBLAS_CALL(cublasDgemm(gpuBlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
                          rowsa, colsb, colsa, &alpha, gpua, rowsa,
                          gpub, rowsb, &beta, gpuc, rowsa));

  blas::getMatrix(rowsa, colsb, sizeof(double), gpuc, rowsa, REAL(ab), rowsa);

  cudaFree(gpua);
  cudaFree(gpub);
  cudaFree(gpuc);

  UNPROTECT(1);
  return ab;
}
