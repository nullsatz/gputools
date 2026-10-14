#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include "cublas.h"
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
void getInverseFromQR(int rows, int cols, const float * dQ, const float * dR,
                      float * dInverse)
{
  float
    * rInverse, * hostIdent;

  if((dQ == NULL) || (dR == NULL) || (dInverse == NULL))
    error("getInverseFromQR: a pointer to a matrix is null");
  if((rows <= 0) || (cols <= 0))
    error("getInverseFromQR: invalid rows or cols argument");

  hostIdent = (float *) R_Calloc(cols * cols, float);
  for(int i = 0; i < cols; i++)
    hostIdent[i + i * cols] = 1.f;

  cublasAlloc(cols * cols, sizeof(float), (void **)&rInverse);
  cublasSetMatrix(cols, cols, sizeof(float), hostIdent, cols, rInverse, cols);
  checkCublasError("getInverseFromQR:");
  R_Free(hostIdent);

  cublasStrsm('L', 'U', 'N', 'N', cols, cols, 1.f, dR, cols, rInverse, cols);
  cublasSgemm('N', 'T', cols, rows, cols, 1.f, rInverse, cols, dQ, rows,
              0.f, dInverse, cols);
  checkCublasError("getInverseFromQR:");
  cublasFree(rInverse);
}

// solves XB=Y for B where X has QR decomp X = QR
// matQ has order rows x cols
// matR has order cols x cols
// vectY has length rows
// vectB has length cols
// please allocate space for vectB before calling solveFromQR
void solveFromQR(int rows, int cols, const float * matQ, const float * matR,
                 const float * vectY,  float * vectB)
{
  float
    * dQ, * dR,
    * dY, * dB,
    * xInverse;

  if((matQ == NULL) || (matR == NULL) || (vectY == NULL) || (vectB == NULL))
    error("solveFromQR: null array argument");
  if((rows <= 0) || (cols <= 0))
    error("solveFromQR: invalid rows or cols argument");

  cublasAlloc(rows * cols, sizeof(float), (void **)&dQ);
  cublasSetMatrix(rows, cols, sizeof(float), matQ, rows, dQ, rows);
  checkCublasError("solveFromQR:");

  cublasAlloc(cols * cols, sizeof(float), (void **)&dR);
  cublasSetMatrix(cols, cols, sizeof(float), matR, cols, dR, cols);
  checkCublasError("solveFromQR:");

  cublasAlloc(rows * cols, sizeof(float), (void **)&xInverse);
  checkCublasError("solveFromQR:");

  getInverseFromQR(rows, cols, dQ, dR, xInverse);
  cublasFree(dQ);
  cublasFree(dR);

  cublasAlloc(rows, sizeof(float), (void **)&dY);
  cublasSetVector(rows, sizeof(float), vectY, 1, dY, 1);
  checkCublasError("solveFromQR:");

  cublasAlloc(cols, sizeof(float), (void **)&dB);
  checkCublasError("solveFromQR:");

  cublasSgemv('N', cols, rows, 1.f, xInverse, cols, dY, 1, 0.f, dB, 1);
  checkCublasError("solveFromQR:");

  cublasFree(xInverse);
  cublasFree(dY);

  cublasGetVector(cols, sizeof(float), dB, 1, vectB, 1);
  checkCublasError("solveFromQR:");
  cublasFree(dB);
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
void getQRDecompRR(int rows, int cols, double tol, float * dQR,
                   int * pivot, double * qrAux, int * rank)
{
  // Copyright 2009, Mark Seligman at Rapid Biologics, LLC.  All rights
  // reserved.
  //
  int
    nblocks, nthreads = NTHREADS,
    fbytes = sizeof(float),
    rowsk, colsk;
  float
    * dColNorms,
    * dV, * dw;

  cublasAlloc(cols, fbytes, (void **) &dColNorms);
  cublasAlloc(rows*cols, fbytes, (void **) &dV);
  cublasAlloc(cols, fbytes, (void**) &dw);
  checkCublasError("getQRDecompRR:");

  // Presets the matrix of Householder vectors, dV, to zero.
  // Padding with zeroes appears to offer better performance than
  // direct operations on submatrices:  see commented-out section
  // in the main loop.
  //
  float
    * zeroes = (float *) R_Calloc(rows*cols, float);
  cublasSetMatrix(rows, cols, fbytes, zeroes, rows, dV, rows);
  R_Free(zeroes);

  checkCublasError("getQRDecompRR:");

  int k;
  double minElt; // Minimum tolerable diagonal element

  nblocks = cols / nthreads;
  if(nblocks * nthreads < cols)
    nblocks++;

  dim3 grid(nblocks), block(nthreads);
  void * argsCN[] = {
    &rows, &cols, &dQR, &rows, &dColNorms
  };
  cudaLaunch("getColNorms", argsCN, grid, block);

  int
    maxRank = rows > cols ? cols : rows;
  float
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
          idx = cublasIsamax(colsk, dColNorms + k, 1) + k - 1;
        if (idx != k) {
          cublasSswap(1, dColNorms + k, 1, dColNorms + idx, 1);
          cublasSswap(rows, pdQRcol, 1, dQR + idx * rows, 1);

          int
            tempIdx = pivot[idx];

          pivot[idx] = pivot[k];
          pivot[k] = tempIdx;
        }
      }

      // Places nonzero elements of V into subdiagonal, although
      // leading element scaled and placed in qrAux.
      //
      cublasScopy(rowsk, pdQRcol + k, 1, pdVdiag, 1);


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
      float
        normV = cublasSnrm2(rowsk, pdVdiag, 1);

      if (k == 0) {
        if (normV < tol) {
          *rank = 0;
          break;
        } else
          minElt = tol * (1.0 + normV);
      } else if (normV < minElt)
        break;

      *rank = k + 1;

      // Builds Householder vector from maximal column just copied.
      // For now, uses slow memory transfers to modify leading element:
      //      V_1 += sign(V_1) * normV.

      float v1;
      cublasGetVector(1, fbytes, pdVdiag, 1, &v1, 1);

      double
        v1Abs = fabs(v1);

      if (rowsk > 1) {  // Scales leading nonzero element of vector.
        double
          fac = 1.0 + normV / v1Abs;
        double recipNormV = 1.0 / normV;
        qrAux[k] = 1.0 + v1Abs * recipNormV;

        cublasSscal(1, (float) fac, pdVdiag, 1);
                  
        // Beta = -2 v^t v :  updates squared norm on host side.

        double beta = -2.0 / (normV*normV + v1Abs * v1Abs * (-1.0 + fac * fac));

        // w = Beta R^t v

        cublasSgemv('T', rows, cols, (float) beta, dQR, rows, pdVcol, 1,
                    0.f, dw, 1);
                                       
        // R = R + v w^t

        cublasSger(rows, cols, 1.0f, pdVcol, 1, dw, 1, dQR, rows);

        // Subdiagonal of V scaled by reciprocal of original signed norm.

        float
          scale = (v1 >= 0.f ? 1.f : -1.f) * recipNormV;
        cublasSscal(rowsk - 1, scale, pdVdiag + 1, 1);

        // Recomputes the squared norms of the remaining columns over the
        // rows still to be reduced; these choose the next pivot.
        int rowsUp = rowsk - 1, colsUp = colsk - 1;
        float
          * trailing = dQR + (k + 1) + (k + 1) * rows,
          * dcnk1 = dColNorms + k + 1;
        void * argsUp[] = {
          &rowsUp, &colsUp, &trailing, &rows, &dcnk1
        };
        cudaLaunch("getColNorms", argsUp, grid, block);
      }
      else
        qrAux[k] = v1Abs;  // The bottom row is not scaled

      checkCublasError("getQRDecompRR:");
      checkCudaError("getQRDecompRR:");
    }

  // Copies the adjusted lower subdiagonal elements into dQR.

  int
    offs = 1;

  for (k = 0; k < *rank; k++, offs += (rows + 1))
    cublasScopy(rows - k - 1, dV + offs, 1, dQR + offs, 1);

  // dQR now contains the upper-triangular portion of the factorization,
  // R.
  // dV is lower-triangular, and contains the Householder vectors, from
  // which the Q portion can be derived.  An adjusted form of the
  // diagonal is saved in qrAux, while the sub-diagonal portion is
  // written onto QR.

  cublasFree(dV);
  cublasFree(dw);
  cublasFree(dColNorms);
        
  checkCublasError("getQRDecompRR:");
}

// use householder xfrms and column pivoting to get the R factor of the
// QR decomp of matrix da:  Q*A*P=R, equiv A*P = Q^t * R
//
void getQRDecompBlocked(int rows, int cols, double tol,
                        float * dQR,
                        int blockSize, int stride, int * pivot,
                        double * qrAux, int * rank)
{
  // Copyright 2009, Mark Seligman at Rapid Biologics, LLC.  All rights
  // reserved.
  //

  int
    nblocks, nthreads = NTHREADS,
    fbytes = sizeof(float),
    rowsk = stride, // # unprocessed rows = stride - k
    colsk = cols,   // # unprocessed columns = cols - k
    maxCol = cols - 1,  // Highest candidate column:  not fixed.
    k = 0; // Number of columns processed.
  const int maxRow = rows - 1;
  float
    * dV, * dW, *du, *dWtR, *dT, *dColNorms;

  checkCublasError("getQRDecompBlocked:");

  nblocks = cols / nthreads;
  if(nblocks * nthreads < cols)
    nblocks++;

  // Presets the matrix of Householder vectors, dV, to zero.
  // Padding with zeroes appears to offer better performance than
  // direct operations on submatrices:  aligned access helps
  // ensure coalescing.
  //
  cublasAlloc(stride * blockSize, fbytes, (void**) &dV);
  cublasAlloc(stride * blockSize, fbytes, (void**) &dW);
  cublasAlloc(blockSize * blockSize, fbytes, (void**) &dT);
  cublasAlloc(stride, fbytes, (void**) &du);
  cublasAlloc(blockSize * (cols - blockSize), fbytes, (void**) &dWtR);
  cublasAlloc(cols, fbytes, (void**) &dColNorms);

  checkCublasError("getQRDecompBlocked allocation:");

  // Obtains the highest valued norm in order to approximate a condition-
  // based lower bound, "minElt".
  //
  dim3 grid(nblocks), block(nthreads);
  void * argsCN[] = {
    &rows, &cols, &dQR, &stride, &dColNorms
  };
  cudaLaunch("getColNorms", argsCN, grid, block);

  int maxIdx = cublasIsamax(cols, dColNorms, 1)-1;
  float maxNorm = cublasSnrm2(rows, dQR + stride * maxIdx, 1);
  int rk = 0; // Local value of rank;
  int maxRank;
  double minElt; // Lowest acceptable norm under given tolerance.

  if (maxNorm < tol)
    maxRank = 0; // Short-circuits the main loop
  else {
    minElt = (1.0 + maxNorm) * tol;
    maxRank = rows > cols ? cols : rows;
  }

  float * pdQRBlock = dQR;

  int blockCount = (cols + blockSize - 1) / blockSize;
  for (int bc = 0; bc < blockCount; bc++) {
    // Determines "blockEnd", which counts the number of columns remaining
    // in the upcoming block.  Swaps trivial columns with the rightmost
    // unvisited column until either a nontrivial column is found or all
    // columns have been visited.  Note that 'blockEnd <= blockSize', with
    // inequality possible only in the rightmost processed block.
    //
    // This pivoting scheme does not attempt to order columns by norm, nor
    // does it recompute norms altered by the rank-one update within the
    // upcoming block.  A higher-fidelity scheme is implemented in the non-
    // blocked form of this function.  Sapienti sat.
    //
    int blockEnd = 0;
    for (int i = k; i < k + blockSize && i < maxRank && i <= maxCol; i++) {
      float colNorm = cublasSnrm2(rows, dQR + i * stride, 1);
      while ( (colNorm < minElt) && (maxCol > i)) {
        cublasSswap(rows, dQR + i * stride, 1, dQR + maxCol*stride, 1);
        int tempIdx = pivot[maxCol];
        pivot[maxCol] = pivot[i];
        pivot[i] = tempIdx;
        maxCol--;
        colNorm = cublasSnrm2(rows, dQR + i * stride, 1);
      }
      if (colNorm >= minElt)
        blockEnd++;
    }
    rk += blockEnd;
    float scales[blockSize];
    double Beta[blockSize];

    cudaMemset2D(dV, blockSize * fbytes, 0.f, blockSize * fbytes, rowsk);

    float *pdVcol = dV;
    float *pdVdiag = dV;
    float *pdQRdiag = pdQRBlock;

    for (int colIdx = 0; colIdx < blockEnd; colIdx++, pdVcol += rowsk, pdVdiag += (rowsk + 1),
           pdQRdiag += (stride + 1), k++) {

      cublasScopy(rowsk - colIdx, pdQRdiag, 1, pdVdiag, 1);

      // Builds Householder vector from maximal column just copied.
      // For now, uses slow memory transfers to modify leading element:
      //      V_1 += sign(V_1) * normV.
      //
      float v1;         
      cublasGetVector(1, fbytes, pdVdiag, rowsk + 1, &v1, 1);
      double v1Abs = fabs(v1);
      if (k == maxRow) // The bottom row is not scaled.
        qrAux[k] = v1Abs;
      else { // zero-valued "normV" should already have been ruled out.
        float normV = cublasSnrm2(rowsk - colIdx, pdQRdiag, 1);
        double recipNormV = 1.0 / normV;
        qrAux[k] = 1.0 + v1Abs * recipNormV;
        scales[colIdx] = (v1 >= 0.f ? 1.f : -1.f) * recipNormV;

        // Scales leading nonzero element of vector.
        //
        double fac = 1.0 + normV / v1Abs;
        cublasSscal(1, (float) fac, pdVdiag, 1);
                  
        // Beta = -2 v^t v :  updates squared norm on host side.
        //
        Beta[colIdx] = -2.0 / (normV*normV + v1Abs * v1Abs * (-1.0 + fac * fac));

        // Rank-one update of the remainder of the block, "B":
        // u = Beta B^t v
        //
        cublasSgemv('T', rowsk, std::min(blockSize,colsk), (float) Beta[colIdx], pdQRBlock, stride, pdVcol, 1, 0.f, du, 1);
                                       
        // B = B + v u^t
        //
        cublasSger(rowsk, std::min(blockSize,colsk), 1.0f, pdVcol, 1, du, 1, pdQRBlock, stride);
      }
    }

    // If more unseen columns remain, updates the remainder of QR lying to
    // the right of the block just updated.  This must be done unless we
    // happen to have exited the inner loop without having applied any
    // Householder transformations (i.e., blockEnd == 0).
    //
    if (bc < blockCount - 1 && blockEnd > 0) {
      // w_m = Beta (I + W V^t) v_m, where the unsubscripted matrices
      // refer to those built at step 'i-1', having 'i' columns.
      //
      // w_i = Beta v_i
      //
      //  T = V^t V
      //
      cublasSsyrk('U', 'T', blockSize, rowsk, 1.f, dV, rowsk, 0.f, dT, blockSize);

      float *pdTcol = dT;
      float *pdWcol = dW;
      pdVcol = dV;
      for (int m = 0; m < blockSize; m++, pdWcol += rowsk, pdVcol += rowsk, pdTcol += blockSize) {
        cublasScopy(rowsk, pdVcol, 1, pdWcol, 1);
        cublasSscal(rowsk, Beta[m], pdWcol, 1);
        // w_m = w_m + Beta W T(.,m)
        //
        if (m > 0) {
          cublasSgemv('N', rowsk, m, Beta[m], dW, rowsk, pdTcol, 1, 1.f, pdWcol, 1);
        }
      }

      // Updates R, beginning at current diagonal by:
      //   R = (I_m + V W^t) R = R + V (W^t R)
      //

      // WtR = W^t R
      //
      cublasSgemm('T','N', blockSize, colsk - blockSize, rowsk, 1.f, dW, rowsk, 
                  pdQRBlock + blockSize * stride, stride, 0.f, dWtR, blockSize);

      // R = V WtR + R
      //
      cublasSgemm('N', 'N', rowsk, colsk - blockSize, blockSize, 1.f, dV,
                  rowsk, dWtR, blockSize, 1.f, pdQRBlock+ blockSize * stride, stride);
    }

    // Flushes scaled Householder vectors to the subdiagonals of dQR,
    // 'blockSize'-many at a time.  The only time a smaller number are
    // sent occurs when a partial block remains at the right end.
    //
    pdVdiag = dV;
    pdQRdiag = pdQRBlock;
    for (int l = 0; l < blockEnd; l++, pdVdiag += (rowsk + 1), pdQRdiag += (stride + 1)) {
      cublasSscal(rowsk - (l + 1), scales[l], pdVdiag + 1, 1);
      cublasScopy(rowsk - (l + 1), pdVdiag + 1, 1, pdQRdiag + 1, 1);
    }

    pdQRBlock += blockSize * (stride + 1);
    colsk -= blockSize;
    rowsk -= blockSize;
  }

  *rank = rk; 
  checkCublasError("getQRDecompBlocked, postblock:");
  checkCudaError("getQRDecompBlocked:");
     
  // dQR now contains the upper-triangular portion of the factorization,
  // R.
  // dV is lower-triangular, and contains the Householder vectors, from
  // which the Q portion can be derived.  An adjusted form of the
  // diagonal is saved in qrAux, while the sub-diagonal portion is
  // written onto QR.

  cublasFree(dT);
  cublasFree(dV);
  cublasFree(dW);
  cublasFree(du);
  cublasFree(dWtR);
  cublasFree(dColNorms);

  checkCublasError("getQRDecompBlocked, freed memory:");
}
