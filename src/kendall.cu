#include "cuseful.h"
#include "R.h"
#include "kendall.h"
#include "nvrtc.h"
#include "cuda.h"

#define NUMTHREADS 16

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      error("\nerror: %d failed with error %s\n", x,              \
            nvrtcGetErrorString(result));                         \
    }                                                             \
  } while(0)

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      error("\nerror: %d failed with error %s\n", x, msg);        \
    }                                                             \
  } while(0)

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

  nvrtcProgram prog;

  NVRTC_SAFE_CALL(
      nvrtcCreateProgram(&prog,  // prog
        kernel_src,              // buffer
        "kendall",               // name
        0,                       // numHeaders
        NULL,                    // headers
        NULL));                  // includeNames

  nvrtcResult compileResult = nvrtcCompileProgram(prog, 0, NULL);
  if (compileResult != NVRTC_SUCCESS) error("cuda kernel compile failed");

  //  Obtain PTX from the program.
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));

  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

  //  Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  //  Load the generated PTX and get a handle to the SAXPY kernel.
//  CUdevice cuDevice;
//  CUcontext context;

  CUDA_SAFE_CALL(cuInit(0));

//  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
//  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

  CUmodule module;
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));

  CUfunction kernel;
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "gpuKendall"));

  // execute kendall kernel
  void *args[] =
    { &gpux
    , &nx
    , &gpuy
    , &ny
    , &sampleSize
    , &gpuResults
    };

  CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
      nx, ny, 1,                  // grid dim
      NUMTHREADS, NUMTHREADS, 1,  // block dim
      0, NULL,                    // shared mem and stream
      args, 0));                  // arguments
  CUDA_SAFE_CALL(cuCtxSynchronize());

  cudaFree(gpux);
  cudaFree(gpuy);
  cudaMemcpy(results, gpuResults, outputBytes, cudaMemcpyDeviceToHost);
  cudaFree(gpuResults);
  checkCudaError("copying results from gpu and cleaning up");

  CUDA_SAFE_CALL(cuModuleUnload(module));
//  CUDA_SAFE_CALL(cuCtxDestroy(context));
}
