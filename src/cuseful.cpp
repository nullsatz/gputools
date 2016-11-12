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

void getComputeNumber(int * major, int * minor)
{
	int currentDevice = 0;
	struct cudaDeviceProp dProps;

	cudaGetDevice(&currentDevice);
	cudaGetDeviceProperties(&dProps, currentDevice);

	*major = dProps.major;
	*minor = dProps.minor;
}

void checkDoubleCapable(const char * failMsg)
{
	int major, minor;
	major = minor = 0;
	getComputeNumber(&major, &minor);
	if((major < 1) || ((major == 1) && (minor < 3)))
		error(failMsg);
}

float * getMatFromFile(int rows, int cols, const char * fn)
{
	FILE * matFile;
	matFile = fopen(fn, "r");
	if(matFile == NULL)
		error("unable to open file %s", fn);
	float * mat = Calloc(rows*cols, float);
	int i, j, err;
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			err = fscanf(matFile, " %f ", mat+i+j*rows);
			if(err == EOF)
				error("file %s incorrect: formatting or size", fn);
		}
		fscanf(matFile, " \n ");
	}
	fclose(matFile);
	return mat;
}

char * getTime() {
	time_t curtime;
	struct tm *loctime;
	curtime = time(NULL);
	loctime = localtime(&curtime);

	return asctime(loctime);
}

void printVect(int n, const float * vect, const char * msg) {
	if(msg != NULL) Rprintf(msg);
	for(int i = 0; i < n; i++) {
		Rprintf("%6.4f, ", vect[i]);
		if((i+1)%10 == 0) Rprintf("\n");
	}
	if(n%10 != 0) Rprintf("\n");
	if(msg != NULL) Rprintf("----------\n");
}

void printMat(int rows, int cols, const float * mat, const char * msg) {
	int i;
	if(msg != NULL) Rprintf(msg);
	for(i = 0; i < rows; i++)
		printVect(cols, mat+i*cols, NULL);
	if(msg != NULL) Rprintf("----------\n");
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

int hasCublasError(const char * msg)
{
	cublasStatus err = cublasGetError();
	if(err != CUBLAS_STATUS_SUCCESS)
		error("cublas error : %s : %s\n", msg, cublasGetErrorString(err).c_str());
	return 0;
}
