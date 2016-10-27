#include "R.h"
#include "nvrtc.h"
#include "cuda.h"
#include "cudaUtils.h"

void cudaCompileLaunch(const char * kernelSrc, const char * kernelName,
    void * args[], int gridDim[3], int blockDim[3])
{
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
      nvrtcCreateProgram(&prog,  // prog
        kernelSrc,               // buffer
        kernelName,              // name
        0,                       // numHeaders
        NULL,                    // headers
        NULL));                  // includeNames

  nvrtcResult compileResult = nvrtcCompileProgram(prog, 0, NULL);
  if (compileResult != NVRTC_SUCCESS) error("cuda kernel compile failed");

  //  Obtain PTX from the program.
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));

  char * ptx = Calloc(ptxSize, char);
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

  //  Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  //  Load the generated PTX and get a handle to the SAXPY kernel.
  CUDA_SAFE_CALL(cuInit(0));

  CUmodule module;
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));

  CUfunction kernel;
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernelName));

  CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
      gridDim[0], gridDim[1], gridDim[2],    // grid dim
      blockDim[0], blockDim[1], blockDim[2], // block dim
      0, NULL,                    // shared mem and stream
      args, 0));                  // arguments
  CUDA_SAFE_CALL(cuCtxSynchronize());

  CUDA_SAFE_CALL(cuModuleUnload(module));
  Free(ptx);
}
