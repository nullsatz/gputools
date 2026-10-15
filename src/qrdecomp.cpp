#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "gpublas.h"
#include "R.h"

#include"cuseful.h"
#include"qrdecomp.h"

#include "cudaUtils.h"

#define NTHREADS 512

// finds inverse for X where X has QR decomp X = QR
// dQ has order rows x cols
// dR has order cols x cols
// please allocate space for dInverse before calling getInverseFromQR
// dQ, dR, and dInverse all live on the gpu device
template <typename T>
void getInverseFromQR(int rows, int cols, const T * dQ, const T * dR,
                      T * dInverse)
{
  T
    * rInverse, * hostIdent;

  if((dQ == NULL) || (dR == NULL) || (dInverse == NULL))
    error("getInverseFromQR: a pointer to a matrix is null");
  if((rows <= 0) || (cols <= 0))
    error("getInverseFromQR: invalid rows or cols argument");

  hostIdent = (T *) R_Calloc(cols * cols, T);
  for(int i = 0; i < cols; i++)
    hostIdent[i + i * cols] = T(1);

  rInverse = gpuAlloc<T>(cols * cols);
  blas::setMatrix(cols, cols, sizeof(T), hostIdent, cols, rInverse, cols);
  R_Free(hostIdent);

  blas::trsmLeftUpper(cols, cols, T(1), dR, cols, rInverse, cols);
  blas::gemm('N', 'T', cols, rows, cols, T(1), rInverse, cols, dQ, rows,
              T(0), dInverse, cols);
  cudaFree(rInverse);
}

// solves XB=Y for B where X has QR decomp X = QR
// matQ has order rows x cols
// matR has order cols x cols
// vectY has length rows
// vectB has length cols
// please allocate space for vectB before calling solveFromQR
template <typename T>
void solveFromQR(int rows, int cols, const T * matQ, const T * matR,
                 const T * vectY,  T * vectB)
{
  T
    * dQ, * dR,
    * dY, * dB,
    * xInverse;

  if((matQ == NULL) || (matR == NULL) || (vectY == NULL) || (vectB == NULL))
    error("solveFromQR: null array argument");
  if((rows <= 0) || (cols <= 0))
    error("solveFromQR: invalid rows or cols argument");

  dQ = gpuAlloc<T>(rows * cols);
  blas::setMatrix(rows, cols, sizeof(T), matQ, rows, dQ, rows);

  dR = gpuAlloc<T>(cols * cols);
  blas::setMatrix(cols, cols, sizeof(T), matR, cols, dR, cols);

  xInverse = gpuAlloc<T>(rows * cols);

  getInverseFromQR(rows, cols, dQ, dR, xInverse);
  cudaFree(dQ);
  cudaFree(dR);

  dY = gpuAlloc<T>(rows);
  blas::setVector(rows, sizeof(T), vectY, 1, dY, 1);

  dB = gpuAlloc<T>(cols);

  blas::gemv('N', cols, rows, T(1), xInverse, cols, dY, 1, T(0), dB, 1);

  cudaFree(xInverse);
  cudaFree(dY);

  blas::getVector(cols, sizeof(T), dB, 1, vectB, 1);
  cudaFree(dB);
}

// implements the Modified Gram-Schmidt QR decomp
// of the matrix A as found in Numerical methods for least squares
// problems by Ake Bjorck
//
// input:  matrix da in gpu memory
//              the matrix is destroyed by the algorithm
// output:  matrices dq and dr, the QR decomp of A
//              dq has dimension rows x cols, dq is orthonormal
//              dr has dimension cols x cols, dr is the upper triangular Cholesky
//                      factor of A^t * A
//              the space for dq and dr should be allocated before calling this
//
// Computes the QR decomposition of the matrix residing in "dQR":
//   Q*A*P=R, equiv A*P = Q^t * R
//
// Employes a rank-revealing coulumn pivot, which will differ appreciably
// from the behavior of R's qr() command.
//
template <typename T>
void getQRDecompRR(int rows, int cols, double tol, T * dQR,
                   int * pivot, double * qrAux, int * rank)
{
  // Copyright 2009, Mark Seligman at Rapid Biologics, LLC.  All rights
  // reserved.
  //
  int
    nblocks, nthreads = NTHREADS,
    fbytes = sizeof(T),
    rowsk, colsk;
  T
    * dColNorms,
    * dV, * dw;

  dColNorms = gpuAlloc<T>(cols);
  dV = gpuAlloc<T>(rows*cols);
  dw = gpuAlloc<T>(cols);

  // Presets the matrix of Householder vectors, dV, to zero.
  // Padding with zeroes appears to offer better performance than
  // direct operations on submatrices:  see commented-out section
  // in the main loop.
  //
  T
    * zeroes = (T *) R_Calloc(rows*cols, T);
  blas::setMatrix(rows, cols, fbytes, zeroes, rows, dV, rows);
  R_Free(zeroes);


  int k;
  double minElt; // Minimum tolerable diagonal element

  nblocks = cols / nthreads;
  if(nblocks * nthreads < cols)
    nblocks++;

  dim3 grid(nblocks), block(nthreads);
  void * argsCN[] = {
    &rows, &cols, &dQR, &rows, &dColNorms
  };
  cudaLaunch("getColNorms", argsCN, grid, block, precisionOf<T>());

  int
    maxRank = rows > cols ? cols : rows;
  T
    * pdVdiag = dV, * pdVcol = dV,
    * pdQRcol = dQR;

  for(k = 0; k < maxRank; k++, pdQRcol  += rows, pdVcol += rows,
        pdVdiag += (rows + 1))
    {
      rowsk = rows - k;
      colsk = cols - k;

      // Obtains zero-based least index of maximum norm, beginning
      // at column 'k'.
      if (colsk > 1) {
        int
          idx = blas::iamax(colsk, dColNorms + k, 1) + k - 1;
        if (idx != k) {
          blas::swap(1, dColNorms + k, 1, dColNorms + idx, 1);
          blas::swap(rows, pdQRcol, 1, dQR + idx * rows, 1);

          int
            tempIdx = pivot[idx];

          pivot[idx] = pivot[k];
          pivot[k] = tempIdx;
        }
      }

      // Places nonzero elements of V into subdiagonal, although
      // leading element scaled and placed in qrAux.
      //
      blas::copy(rowsk, pdQRcol + k, 1, pdVdiag, 1);


      // This should probably be moved to the device.
      //
      // Determines rank, using tolerance to bound condition number.
      // Pivoting has placed the diagonal elements in decreasing order,
      // with the stronger property that the ith. diagonal element has
      // higher magnitude than the 2-norm of the upper i+1st. column
      // (cf. Bjorck).
      //
      // N.B.:  pivoting is not a foolproof way to compute rank, however.
      //
      // Tolerance should be positve.
      //
      T
        normV = blas::nrm2(rowsk, pdVdiag, 1);

      // Bjorck's rank test, |r_kk| < tol |r_11|: relative, so the rank
      // does not depend on the units of the data (it used
      // tol (1 + |r_11|), which declared data in small units rank 0)
      if (k == 0) {
        if (normV == T(0)) {
          *rank = 0;
          break;
        } else
          minElt = tol * normV;
      } else if (normV < minElt)
        break;

      *rank = k + 1;

      // Builds Householder vector from maximal column just copied.
      // For now, uses slow memory transfers to modify leading element:
      //      V_1 += sign(V_1) * normV.

      T v1;
      blas::getVector(1, fbytes, pdVdiag, 1, &v1, 1);

      double
        v1Abs = fabs(v1);

      if (rowsk > 1) {  // Scales leading nonzero element of vector.
        double
          fac = 1.0 + normV / v1Abs;
        double recipNormV = 1.0 / normV;
        qrAux[k] = 1.0 + v1Abs * recipNormV;

        blas::scal(1, (T) fac, pdVdiag, 1);
                  
        // Beta = -2 v^t v :  updates squared norm on host side.

        double beta = -2.0 / (normV*normV + v1Abs * v1Abs * (-1.0 + fac * fac));

        // w = Beta R^t v

        blas::gemv('T', rows, cols, (T) beta, dQR, rows, pdVcol, 1,
                    T(0), dw, 1);
                                       
        // R = R + v w^t

        blas::ger(rows, cols, T(1.0), pdVcol, 1, dw, 1, dQR, rows);

        // Subdiagonal of V scaled by reciprocal of original signed norm.

        T
          scale = (v1 >= T(0) ? T(1) : -T(1)) * recipNormV;
        blas::scal(rowsk - 1, scale, pdVdiag + 1, 1);

        // Recomputes the squared norms of the remaining columns over the
        // rows still to be reduced; these choose the next pivot.
        int rowsUp = rowsk - 1, colsUp = colsk - 1;
        T
          * trailing = dQR + (k + 1) + (k + 1) * rows,
          * dcnk1 = dColNorms + k + 1;
        void * argsUp[] = {
          &rowsUp, &colsUp, &trailing, &rows, &dcnk1
        };
        cudaLaunch("getColNorms", argsUp, grid, block, precisionOf<T>());
      }
      else
        qrAux[k] = v1Abs;  // The bottom row is not scaled

      checkCudaError("getQRDecompRR:");
    }

  // Copies the adjusted lower subdiagonal elements into dQR.

  int
    offs = 1;

  for (k = 0; k < *rank; k++, offs += (rows + 1))
    blas::copy(rows - k - 1, dV + offs, 1, dQR + offs, 1);

  // dQR now contains the upper-triangular portion of the factorization,
  // R.
  // dV is lower-triangular, and contains the Householder vectors, from
  // which the Q portion can be derived.  An adjusted form of the
  // diagonal is saved in qrAux, while the sub-diagonal portion is
  // written onto QR.

  cudaFree(dV);
  cudaFree(dw);
  cudaFree(dColNorms);
        
}

// QR decomposition as LINPACK's dqrdc2, which lm() and lsfit() use, so
// that gpuLm/gpuLsfit/gpuGlm detect rank deficiency the way they do:
// columns are taken in order, and one whose norm below the rows already
// reduced has fallen under tol times its original norm is moved to the
// end (the rest keep their order). Every column is still reduced, so the
// result is LINPACK's compact form: R on and above the diagonal, the
// Householder vectors below it with their leading elements in qrAux.
// rank is the number of columns not moved.
//
// dQR (rows x cols, leading dimension rows) is on the device; pivot
// (0-based column indices) and qrAux are on the host.
template <typename T>
void getQRDecompLinpack(int rows, int cols, double tol, T * dQR,
                        int * pivot, double * qrAux, int * rank)
{
  const int fbytes = sizeof(T), lup = std::min(rows, cols);
  std::vector<double> origNorm(cols);
  for (int j = 0; j < cols; ++j) {
    origNorm[j] = blas::nrm2(rows, dQR + (size_t) j * rows, 1);
    if (origNorm[j] == 0.0) origNorm[j] = 1.0;  // as dqrdc2's work(j,2)
  }

  T * dw = gpuAlloc<T>(cols), * dTmp = gpuAlloc<T>(rows);
  int kept = cols;  // columns [kept, cols) have been moved to the end

  for (int l = 0; l < lup; ++l) {
    T * col = dQR + (size_t) l * rows + l;
    int m = rows - l;
    double remaining = blas::nrm2(m, col, 1);

    // move negligible columns to the end, keeping the others' order
    while (l < kept && remaining < origNorm[l] * tol) {
      T * colTop = dQR + (size_t) l * rows;
      CUDART_CALL(cudaMemcpy(dTmp, colTop, rows * fbytes,
                             cudaMemcpyDeviceToDevice));
      for (int j = l + 1; j < cols; ++j)
        CUDART_CALL(cudaMemcpy(dQR + (size_t) (j - 1) * rows,
                               dQR + (size_t) j * rows, rows * fbytes,
                               cudaMemcpyDeviceToDevice));
      CUDART_CALL(cudaMemcpy(dQR + (size_t) (cols - 1) * rows, dTmp,
                             rows * fbytes, cudaMemcpyDeviceToDevice));
      std::rotate(pivot + l, pivot + l + 1, pivot + cols);
      std::rotate(origNorm.begin() + l, origNorm.begin() + l + 1,
                  origNorm.end());
      --kept;
      remaining = blas::nrm2(m, col, 1);
    }

    if (l == rows - 1) {          // the last row: nothing to reflect
      qrAux[l] = remaining;
      continue;
    }
    if (remaining == 0.0) {
      qrAux[l] = 0.0;
      continue;
    }

    // Householder reflection of column l, LINPACK's formulation
    T xll;
    blas::getVector(1, fbytes, col, 1, &xll, 1);
    T nrmxl = (T) remaining;
    if (xll != T(0) && xll < T(0)) nrmxl = -nrmxl;
    blas::scal(m, T(1) / nrmxl, col, 1);
    blas::getVector(1, fbytes, col, 1, &xll, 1);
    xll += T(1);
    blas::setVector(1, fbytes, &xll, 1, col, 1);

    int right = cols - l - 1;
    if (right > 0) {
      // x_j -= (u . x_j / u_l) u for every column j to the right
      T * rest = dQR + (size_t) (l + 1) * rows + l;
      blas::gemv('T', m, right, T(1), rest, rows, col, 1, T(0), dw, 1);
      blas::scal(right, T(-1) / xll, dw, 1);
      blas::ger(m, right, T(1), col, 1, dw, 1, rest, rows);
    }
    qrAux[l] = xll;
    T rll = -nrmxl;
    blas::setVector(1, fbytes, &rll, 1, col, 1);
  }

  *rank = std::min(kept, rows);
  cudaFree(dw);
  cudaFree(dTmp);
}

template void getInverseFromQR<float>(int rows, int cols, const float * dQ, const float * dR,
                      float * dInverse);
template void getInverseFromQR<double>(int rows, int cols, const double * dQ, const double * dR,
                      double * dInverse);
template void solveFromQR<float>(int rows, int cols, const float * matQ, const float * matR,
                 const float * vectY,  float * vectB);
template void solveFromQR<double>(int rows, int cols, const double * matQ, const double * matR,
                 const double * vectY,  double * vectB);
template void getQRDecompRR<float>(int rows, int cols, double tol, float * dQR,
                   int * pivot, double * qrAux, int * rank);
template void getQRDecompRR<double>(int rows, int cols, double tol, double * dQR,
                   int * pivot, double * qrAux, int * rank);
template void getQRDecompLinpack<float>(int rows, int cols, double tol,
  float * dQR, int * pivot, double * qrAux, int * rank);
template void getQRDecompLinpack<double>(int rows, int cols, double tol,
  double * dQR, int * pivot, double * qrAux, int * rank);
