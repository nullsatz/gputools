#include <map>
#include <vector>
#include <string>

#include "R.h"
#include "nvrtc.h"
#include "cuda.h"

#include "cudaUtils.h"

std::map<std::string, const char *> * cudaKernels;
std::vector<char *> ptxToFree;

// Obtain compilation log from the program.
void printCompileLog(nvrtcProgram &prog) {
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char * log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  warning(log);
  delete[] log;
}

static
std::vector<std::string> & getFileKernels(std::string file)
{
  std::vector<std::string> * kernels;
  if (file == "correlation") {  
    std::string newKernels[] =
      { "gpuSignif"
      , "gpuMeans"
      , "gpuSD"
      , "gpuPMCC"
      , "gpuMeansNoTest"
      , "gpuSDNoTest"
      , "gpuPMCCNoTest"
      , "dUpdateSignif"
      , "noNAsPmccMeans"
      };
    kernels = new std::vector<std::string>(newKernels, newKernels + 9);
  } else if (file == "distance") {
    std::string newKernels[] = 
      { "euclidean_kernel_same"
      , "maximum_kernel_same"
      , "manhattan_kernel_same"
      , "canberra_kernel_same"
      , "binary_kernel_same"
      , "minkowski_kernel_same"
      };
    kernels = new std::vector<std::string>(newKernels, newKernels + 6);
  } else if (file == "granger") {
    std::string newKernels[] = 
      { "getRestricted"
      , "getUnrestricted"
      , "ftest"
      , "getRestricted"
      , "getUnrestricted"
      , "ftest"
      };
    kernels = new std::vector<std::string>(newKernels, newKernels + 6);
  } else if (file == "hcluster") {
    std::string newKernels[] = 
      { "complete_kernel"
      , "wpgma_kernel"
      , "average_kernel"
      , "median_kernel"
      , "centroid_kernel"
      , "flexible_group_kernel"
      , "flexible_kernel"
      , "ward_kernel"
      , "mcquitty_kernel"
      , "single_kernel"
      , "convert_kernel"
      , "find_min1_kernel"
      , "find_min2_kernel"
      };
    kernels = new std::vector<std::string>(newKernels, newKernels + 13);
  } else if (file == "kendall") {
    std::string newKernels[] = { "gpuKendall" };
    kernels = new std::vector<std::string>(newKernels, newKernels + 1);
  } else if (file == "mi") {
    std::string newKernels[] =
      { "scale"
      , "get_bin_scores"
      , "get_entropy"
      , "get_mi"
      };
    kernels = new std::vector<std::string>(newKernels, newKernels + 4);
  } else if (file == "qrdecomp") {
    std::string newKernels[] =
      { "getColNorms"
      , "gpuFindMax"
      , "gpuSwapCol"
      , "makeHVector"
      , "UpdateHHNorms"
      };
    kernels = new std::vector<std::string>(newKernels, newKernels + 5);
  } else {
    kernels = new std::vector<std::string>();
  }
  return(*kernels);
}

extern "C"
void cuCompile(const int * numFiles,
               const char ** cuFilenames,
               const char ** cuSrc)
{
  cudaKernels = new std::map<std::string, const char *>();
  CUDA_SAFE_CALL(cuInit(0));

  for (int i = 0; i < *numFiles; ++i) {
    std::string file = cuFilenames[i];
    const char * src = cuSrc[i];
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(
      nvrtcCreateProgram(&prog,  // prog
        src,                     // buffer
        file.c_str(),            // name
        0,                       // numHeaders
        NULL,                    // headers
        NULL));                  // includeNames

    const char * options[] =
    { "--use_fast_math"
//    , "--gpu-architecture"
//    , "compute_30"
    };

    nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, options);
    if (compileResult != NVRTC_SUCCESS) {
      printCompileLog(prog);
      error("\ncuda kernel compile failed");
    }

    //  Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));

    char * ptx = Calloc(ptxSize, char);
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

    //  Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    std::vector<std::string> kernels = getFileKernels(file);
    for(int i = 0; i < kernels.size(); ++i) {
      (*cudaKernels)[kernels[i]] = ptx;
      ptxToFree.push_back(ptx);
    }
  }
}

extern "C"
void unloadPackage()
{
  for(int i = 0; i < ptxToFree.size(); ++i) {
    Free(ptxToFree[i]);
  }
  delete cudaKernels;
}

void cudaLaunch(std::string kernelName,
                void * args[],
                const dim3 &gridDim, const dim3 &blockDim,
                cudaStream_t stream)
{
  const char * ptx = (*cudaKernels)[kernelName];

  CUmodule module;
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));

  CUfunction kernel;
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernelName.c_str()));

  CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
      gridDim.x, gridDim.y, gridDim.z,    // grid dim
      blockDim.x, blockDim.y, blockDim.z, // block dim
      0, stream,                    // shared mem and stream
      args, 0));                  // arguments
  //  CUDA_SAFE_CALL(cuCtxSynchronize());
  CUDA_SAFE_CALL(cuModuleUnload(module));
}
