#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<cuda_runtime.h>
#include<cublas_v2.h>

#include<cuseful.h>
#include<matmult.h>

RcppExport SEXP gpuMatMult(SEXP a, SEXP b) {
	Rcpp::NumericMatrix xa(a);
	Rcpp::NumericMatrix xb(b);

	double
		* gpua, * gpub, * gpuc;

	int
		rowsa = xa.nrow(), colsa = xa.ncol(),
		rowsb = xb.nrow(), colsb = xb.ncol();

	cublasStatus_t stat;
	cublasHandle_t handle;

	cudaError_t cudaStat;

	cudaStat = cudaMalloc((void**) &gpua, rowsa * colsa * sizeof(double));
	if (cudaStat != cudaSuccess) {
		printf ("device memory allocation failed");
		return NULL;
	}  

	cudaStat = cudaMalloc((void**) &gpub, rowsb * colsb * sizeof(double));
	if (cudaStat != cudaSuccess) {
		printf ("device memory allocation failed");
		return NULL;
	}  

	cublasOperation_t opA = tpA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t opB = tpB ? CUBLAS_OP_T : CUBLAS_OP_N;

	int rowsOpA = tpA ? colsa : rowsa;
	int colsOpA = tpA ? rowsa : colsa;
	int colsOpB = tpB ? rowsb : colsb;

	cudaStat = cudaMalloc((void**) &gpuc, rowsOpA * colsOpB * sizeof(double));
	if (cudaStat != cudaSuccess) {
		printf ("device memory allocation failed");
		return NULL;
	}  

	stat = cublasCreate(&handle);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed\n");
		return NULL;
	}

	stat = cublasSetMatrix(rowsa, colsa, sizeof(double), xa, rowsa,
		gpua, rowsa);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf("data download failed\n");
		cudaFree(gpuc);
		cudaFree(gpub);
		cudaFree(gpua);
		cublasDestroy(handle);
		return NULL;
	}

	Rcpp::NumericMatrix xb(b);
	stat = cublasSetMatrix(rowsb, colsb, sizeof(double), xb, rowsb,
		gpub, rowsb);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf("data download failed\n");
		cudaFree(gpuc);
		cudaFree(gpub);
		cudaFree(gpua);
		cublasDestroy(handle);
		return NULL;
	}

	const double alpha = 1.0, beta = 0.0;
	cublasDgemm(handle, opA, opB, rowsOpA, colsOpB, colsOpA, &alpha,
		(const double *) gpua, rowsa, (const double *) gpub, rowsb,
		&beta, gpuc, rowsOpA);

	Rcpp::NumericMatrix xab(rowsa, colsb);
	stat = cublasGetMatrix(rowsOpA, colsOpB, sizeof(double), gpuc, rowsOpA,
		xab, rowsOpA);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf("data upload failed\n");
		cudaFree(gpuc);
		cudaFree(gpub);
		cudaFree(gpua);
		cublasDestroy(handle);
		return NULL;
	}

	cudaFree(gpua);
	cudaFree(gpub);
	cudaFree(gpuc);

	cublasDestroy(handle);
	return xab;
}
