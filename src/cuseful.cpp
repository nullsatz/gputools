#include "cuda_runtime.h"

#include "R.h"

#include "cuseful.h"

void fatal(const char * msg)
{
	error("%s", msg);
}

int hasCudaError(const char * msg) {
	checkCudaError(msg);
	return 0;
}

void checkCudaError(const char * msg) {
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err)
		error("cuda error : %s : %s", msg ? msg : "", cudaGetErrorString(err));
}
