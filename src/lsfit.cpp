#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include "gpublas.h"

#include "R.h"

#include "cuseful.h"
#include "lsfit.h"
#include "qrdecomp.h"

// Copyright 2009, Mark Seligman at Rapid Biologics, LLC.  All rights
// reserved.
//

template <typename T>
void leastSquaresFit(T * X, int rows, int cols, T * Y, int yCols,
               double tol, T * coeffs, T * resids, T * effects,
               int * rank, int * pivot, double * qrAux)
{
	const int
		fbytes = sizeof(T), stride = rows;

	T * dQR = gpuAlloc<T>((size_t) rows * cols);
	blas::setMatrix(rows, cols, fbytes, X, rows, dQR, stride);

	// dQR in pivoted, packed QR form, as lm()'s dqrdc2 leaves it
	getQRDecompLinpack(rows, cols, tol, dQR, pivot, qrAux, rank);
	blas::getMatrix(rows, cols, fbytes, dQR, stride, X, rows);

	if(*rank > 0)
		getCRE(dQR, rows, cols, stride, *rank, qrAux, yCols, coeffs, resids, effects);
	else // Residuals copied from Y.
		memcpy(resids, Y, rows * yCols * fbytes);

   	cudaFree(dQR);
}

// Fills in the coefficients, residuals and effects matrices.
//
template <typename T>
void getCRE(T *dQR, int rows, int cols, int stride, int rank, double *qrAux,
	int yCols, T *coeffs, T *resids, T *effects)
{
	const int
		fbytes = sizeof(T);
        // Used by effects, residual computations.
	//
	int maxIdx = std::min(rank, rows - 1);

	T
		* diags = R_Calloc(rank * fbytes, T),
		* dDiags, *dResids, *dCoeffs, *dEffects;

	dDiags = gpuAlloc<T>(rank);

	dCoeffs = gpuAlloc<T>(cols * yCols);
	dResids = gpuAlloc<T>(rows * yCols);
	dEffects = gpuAlloc<T>(rows * yCols);

	// Temporarily swaps diagonals with qrAux.

	blas::copy(rank, dQR, stride + 1, dDiags, 1);
	blas::getVector(rank, fbytes, dDiags, 1, diags, 1);

	T *qrAuxT = R_Calloc(maxIdx * fbytes, T);
	for (int i = 0; i < maxIdx; i++)
	  qrAuxT[i] = qrAux[i];
	blas::setVector(maxIdx, fbytes, qrAuxT, 1, dQR, stride + 1);
	R_Free(qrAuxT);

	blas::setMatrix(cols, yCols, fbytes, coeffs, cols, dCoeffs, cols);
	blas::setMatrix(rows, yCols, fbytes, effects, rows, dEffects, rows);
	blas::setMatrix(rows, yCols, fbytes, resids, rows, dResids, rows);

	// Computes the effects matrix, intialized by caller to Y.

	T
		* pEffects = dEffects;

	for (int i = 0; i < yCols; i++, pEffects += rows) {
		T
			* pQR = dQR;

		for (int k = 0; k < maxIdx; k++, pQR += (stride + 1)) {
			double
				t = blas::dot(rows - k, pQR, 1, pEffects +  k, 1);

			t *= -1.0 / qrAux[k];
			blas::axpy(rows - k, t, pQR, 1, pEffects + k, 1);
		}
	}

	// Computes the residuals matrix, initialized by caller to zero.
	// If not of full row rank, presets the remaining rows to those from
	// effects.

	if(rank < rows) {
		for(int i = 0; i < yCols; i++) {
			blas::copy(rows - rank,  dEffects + i*rows + rank, 1,
				dResids + i*rows + rank, 1);
		}
	}

	T
		* pResids = dResids;

	for (int i = 0; i < yCols; i++, pResids += rows) {
		for (int k = maxIdx - 1; k >= 0; k--) {
			double
				t = -(1.0 / qrAux[k])
					* blas::dot(rows - k, dQR + k*stride + k, 1, pResids + k, 1);

			blas::axpy(rows -k, t, dQR + k*stride + k, 1, pResids + k, 1);
		}
	}
	blas::copy(maxIdx, dDiags, 1, dQR, stride + 1);

	// Computes the coefficients matrix, initialized by caller to zero.

	T
		* pCoeffs = dCoeffs;

	for(int i = 0; i < yCols; i++, pCoeffs += cols) {
		blas::copy(rank, dEffects + i*rows, 1, pCoeffs, 1);

		T t;
		for(int k = rank - 1; k > 0; k--) {
		        blas::scal(1, T(1) / diags[k], pCoeffs + k, 1);
			blas::getVector(1, fbytes, pCoeffs + k, 1, &t, 1);
			blas::axpy(k, -t, dQR + k*stride, 1, pCoeffs, 1);
		}
		blas::scal(1, T(1) / diags[0], pCoeffs, 1);
	}
	R_Free(diags);

	blas::getMatrix(cols, yCols, fbytes, dCoeffs, cols, coeffs, cols);
	blas::getMatrix(rows, yCols, fbytes, dResids, rows, resids, rows);
	blas::getMatrix(rows, yCols, fbytes, dEffects, rows, effects, rows);

	cudaFree(dDiags);
	cudaFree(dCoeffs);
	cudaFree(dResids);
	cudaFree(dEffects);
}

template void leastSquaresFit<float>(float * X, int rows, int cols, float * Y, int yCols,
               double tol, float * coeffs, float * resids, float * effects,
               int * rank, int * pivot, double * qrAux);
template void leastSquaresFit<double>(double * X, int rows, int cols, double * Y, int yCols,
               double tol, double * coeffs, double * resids, double * effects,
               int * rank, int * pivot, double * qrAux);
template void getCRE<float>(float *dQR, int rows, int cols, int stride, int rank, double *qrAux,
	int yCols, float *coeffs, float *resids, float *effects);
template void getCRE<double>(double *dQR, int rows, int cols, int stride, int rank, double *qrAux,
	int yCols, double *coeffs, double *resids, double *effects);
