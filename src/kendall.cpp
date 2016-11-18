#include "nvrtc.h"
#include "cuda.h"

#include "R.h"

#include "cuseful.h"
#include "cudaUtils.h"

#include "kendall.h"

#define NUMTHREADS 16

void masterKendall(const float * x, size_t nx, 
                   const float * y, size_t ny,
                   size_t sampleSize, double * results)
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
  cudaLaunch("gpuKendall", args,
      grid, block);

  cudaFree(gpux);
  cudaFree(gpuy);
  cudaMemcpy(results, gpuResults, outputBytes, cudaMemcpyDeviceToHost);
  cudaFree(gpuResults);
  checkCudaError("copying results from gpu and cleaning up");
}
