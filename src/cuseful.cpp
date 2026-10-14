#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<cublas.h>
#include<R.h>
#include<Rinternals.h>

#include"cuseful.h"

#include<string>

#define HALF RAND_MAX/2

void fatal(const char * msg)
{
	error(msg);
}

int hasCudaError(const char * msg) {
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err)
		error("cuda error : %s : %s\n", msg, cudaGetErrorString(err));
	return 0;
}

void checkCudaError(const char * msg) {
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) {
		if(msg != NULL)
			warning(msg);
		error(cudaGetErrorString(err));
	}
}

std::string cublasGetErrorString(cublasStatus err)
{
	switch(err) {
		case CUBLAS_STATUS_SUCCESS :
			return "operation completed successfully";
		case CUBLAS_STATUS_NOT_INITIALIZED :
			return "CUBLAS library not initialized";
		case CUBLAS_STATUS_ALLOC_FAILED :
			return "resource allocation failed";
		case CUBLAS_STATUS_INVALID_VALUE :
			return "unsupported numerical value was passed to function";
		case CUBLAS_STATUS_ARCH_MISMATCH :
			return "function requires an architectural feature absent from \
			the architecture of the device";
		case CUBLAS_STATUS_MAPPING_ERROR :
			return "access to GPU memory space failed";
		case CUBLAS_STATUS_EXECUTION_FAILED :
			return "GPU program failed to execute";
		case CUBLAS_STATUS_INTERNAL_ERROR :
			return "an internal CUBLAS operation failed";
		default :
			return "unknown error type";
	}
}

void checkCublasError(const char * msg)
{
	cublasStatus err = cublasGetError();
	if(err != CUBLAS_STATUS_SUCCESS)
		error("cublas error : %s : %s\n", msg, cublasGetErrorString(err).c_str());
}

