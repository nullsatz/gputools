#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include "cublas.h"

#include "R.h"

#include "cuseful.h"
#include "lsfit.h"
#include "qrdecomp.h"

// Copyright 2009, Mark Seligman at Rapid Biologics, LLC.  All rights
// reserved.
//

// Rounds "length" up to the next multiple of the block length.
//
int alignBlock(int length, unsigned blockExp) {
  int blockSize = 1 << blockExp;
  return (length + blockSize - 1) & (((unsigned) -1)  << blockExp);
}

void gpuLSFitF(float * X, int rows, int cols, float * Y, int yCols,
               double tol, float * coeffs, float * resids, float * effects,
               int * rank, int * pivot, double * qrAux)
{
	const int
		fbytes = sizeof(float);

	// Should be >= 4, to satisfy alignment criterea for memory
	// coalescence.  For larger arrays (> 1000 rows), best performance
	// has been observed at 7.
	//
	const unsigned blockExp = 7; // Gives blockSize = 2^7 = 128.

	float *dQR;

	int stride = alignBlock(rows, blockExp);
	
	cublasInit();
	cublasAlloc(stride * cols, fbytes, (void **)&dQR);

	// This is overkill:  just need to zero the padding.
	//
	cudaMemset2D(dQR, cols * fbytes, 0.f, cols * fbytes, stride); 
	cublasSetMatrix(rows, cols, fbytes, X, rows, dQR, stride);

  // On return we have dQR in pivoted, packed QR form.
  
  getQRDecompBlocked(rows, cols, tol, dQR, 1 << blockExp,
                     stride, pivot, qrAux, rank);
	cublasGetMatrix(rows, cols, fbytes, dQR, stride, X, rows);

	if(*rank > 0)
		getCRE(dQR, rows, cols, stride, *rank, qrAux, yCols, coeffs, resids, effects);
	else // Residuals copied from Y.
		memcpy(resids, Y, rows * yCols * fbytes);

   	cublasFree(dQR);
	cublasShutdown();
}

void gpuLSFitD(double *X, int n, int p, double *Y, int nY,
	   double tol, double *coeffs, double *resids, double *effects,
	   int *rank, int *pivot, double * qrAux) 
{
// NYI
}


// Fills in the coefficients, residuals and effects matrices.
//
void getCRE(float *dQR, int rows, int cols, int stride, int rank, double *qrAux,
	int yCols, float *coeffs, float *resids, float *effects)
{
	const int
		fbytes = sizeof(float);
        // Used by effects, residual computations.
	//
	int maxIdx = std::min(rank, rows - 1);

	float
		* diags = Calloc(rank * fbytes, float),
		* dDiags, *dResids, *dCoeffs, *dEffects;

	cublasAlloc(rank, fbytes, (void **) &dDiags);

	cublasAlloc(cols * yCols, fbytes, (void **) &dCoeffs);
	cublasAlloc(rows * yCols, fbytes, (void **) &dResids);
	cublasAlloc(rows * yCols, fbytes, (void **) &dEffects);

	// Temporarily swaps diagonals with qrAux.

	cublasScopy(rank, dQR, stride + 1, dDiags, 1);
	cublasGetVector(rank, fbytes, dDiags, 1, diags, 1);

	float *qrAuxFloat = Calloc(maxIdx * fbytes, float);
	for (int i = 0; i < maxIdx; i++)
	  qrAuxFloat[i] = qrAux[i];
	cublasSetVector(maxIdx, fbytes, qrAuxFloat, 1, dQR, stride + 1);
	Free(qrAuxFloat);

	cublasSetMatrix(cols, yCols, fbytes, coeffs, cols, dCoeffs, cols);
	cublasSetMatrix(rows, yCols, fbytes, effects, rows, dEffects, rows);
	cublasSetMatrix(rows, yCols, fbytes, resids, rows, dResids, rows);

	// Computes the effects matrix, intialized by caller to Y.

	float
		* pEffects = dEffects;

	for (int i = 0; i < yCols; i++, pEffects += rows) {
		float
			* pQR = dQR;

		for (int k = 0; k < maxIdx; k++, pQR += (stride + 1)) {
			double
				t = cublasSdot(rows - k, pQR, 1, pEffects +  k, 1);

			t *= -1.0 / qrAux[k];
			cublasSaxpy(rows - k, t, pQR, 1, pEffects + k, 1);
		}
	}

	// Computes the residuals matrix, initialized by caller to zero.
	// If not of full row rank, presets the remaining rows to those from
	// effects.

	if(rank < rows) {
		for(int i = 0; i < yCols; i++) {
			cublasScopy(rows - rank,  dEffects + i*rows + rank, 1,
				dResids + i*rows + rank, 1);
		}
	}

	float
		* pResids = dResids;

	for (int i = 0; i < yCols; i++, pResids += rows) {
		for (int k = maxIdx - 1; k >= 0; k--) {
			double
				t = -(1.0 / qrAux[k])
					* cublasSdot(rows - k, dQR + k*stride + k, 1, pResids + k, 1);

			cublasSaxpy(rows -k, t, dQR + k*stride + k, 1, pResids + k, 1);
		}
	}
	cublasScopy(maxIdx, dDiags, 1, dQR, stride + 1);

	// Computes the coefficients matrix, initialized by caller to zero.

	float
		* pCoeffs = dCoeffs;

	for(int i = 0; i < yCols; i++, pCoeffs += cols) {
		cublasScopy(rank, dEffects + i*rows, 1, pCoeffs, 1);

		float t;
		for(int k = rank - 1; k > 0; k--) {
		        cublasSscal(1, 1.f / diags[k], pCoeffs + k, 1);
			cublasGetVector(1, fbytes, pCoeffs + k, 1, &t, 1);
			cublasSaxpy(k, -t, dQR + k*stride, 1, pCoeffs, 1);
		}
		cublasSscal(1, 1.f / diags[0], pCoeffs, 1);
	}
	Free(diags);

	cublasGetMatrix(cols, yCols, fbytes, dCoeffs, cols, coeffs, cols);
	cublasGetMatrix(rows, yCols, fbytes, dResids, rows, resids, rows);
	cublasGetMatrix(rows, yCols, fbytes, dEffects, rows, effects, rows);

	cublasFree(dDiags);
	cublasFree(dCoeffs);
	cublasFree(dResids);
	cublasFree(dEffects);
}
