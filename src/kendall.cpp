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
		grid(nx, ny), block(NUMTHREADS, NUMTHREADS);

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
  cudaCompileLaunch(kernel_src, "gpuKendall", args,
      grid, block);

  cudaFree(gpux);
  cudaFree(gpuy);
  cudaMemcpy(results, gpuResults, outputBytes, cudaMemcpyDeviceToHost);
  cudaFree(gpuResults);
  checkCudaError("copying results from gpu and cleaning up");
}
