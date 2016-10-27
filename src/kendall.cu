#include "cuseful.h"
#include "R.h"
#include "kendall.h"
#include "nvrtc.h"
#include "cuda.h"
#include "cudaUtils.h"

#define NUMTHREADS 16

void masterKendall(const float * x,  size_t nx, 
  const float * y, size_t ny,
  size_t sampleSize, double * results,
  const char * kernel_src)
{
	size_t 
		outputLength = nx * ny, outputBytes = outputLength*sizeof(double),
		xBytes = nx*sampleSize*sizeof(float), 
		yBytes = ny*sampleSize*sizeof(float); 
	float
		* gpux, * gpuy; 
	double
		* gpuResults;
	dim3
		initGrid(nx, ny), initBlock(NUMTHREADS, NUMTHREADS);

	cudaMalloc((void **)&gpux, xBytes);
	cudaMalloc((void **)&gpuy, yBytes);
	checkCudaError("input vector space allocation");

	cudaMemcpy(gpux, x, xBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuy, y, yBytes, cudaMemcpyHostToDevice);
	checkCudaError("copying input vectors to gpu");

	cudaMalloc((void **)&gpuResults, outputBytes);
	checkCudaError("allocation of space for result matrix");

  void *args[] =
    { &gpux
    , &nx
    , &gpuy
    , &ny
    , &sampleSize
    , &gpuResults
    };
  int
    gridDim[3] = {nx, ny, 1},
    blockDim[3] = {NUMTHREADS, NUMTHREADS, 1};
  cudaCompileLaunch(kernel_src, "gpuKendall", args,
      gridDim, blockDim); 

  cudaFree(gpux);
  cudaFree(gpuy);
  cudaMemcpy(results, gpuResults, outputBytes, cudaMemcpyDeviceToHost);
  cudaFree(gpuResults);
  checkCudaError("copying results from gpu and cleaning up");
}
