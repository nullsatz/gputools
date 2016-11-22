#include <map>
#include <vector>
#include <string>
#include <algorithm>

#include "R.h"
#include "nvrtc.h"
#include "cuda.h"

#include "cudaUtils.h"

class CudaKernel {
public:
  const char * name;
  const char * ptx;
  nvrtcProgram * prog;

  CudaKernel(const char * _name, const char * _ptx, nvrtcProgram * _prog)
  {
    name = _name;
    ptx = _ptx;
    prog = _prog;
  }
};

std::map<std::string, CudaKernel *> * cudaKernels;

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
  cudaKernels = new std::map<std::string, CudaKernel *>();
  CUDA_SAFE_CALL(cuInit(0));

  for (int i = 0; i < *numFiles; ++i) {
    std::string file = cuFilenames[i];
    const char * src = cuSrc[i];

    nvrtcProgram * prog = new nvrtcProgram();
    NVRTC_SAFE_CALL(
      nvrtcCreateProgram(prog,   // prog
        src,                     // buffer
        file.c_str(),            // name
        0,                       // numHeaders
        NULL,                    // headers
        NULL));                  // includeNames
    
    std::vector<std::string> kernels = getFileKernels(file);
    for(int i = 0; i < kernels.size(); ++i) {
      NVRTC_SAFE_CALL(nvrtcAddNameExpression(*prog, kernels[i].c_str()));
    }
    
    const char * options[] = { "--use_fast_math" };

    nvrtcResult compileResult = nvrtcCompileProgram(*prog, 1, options);
    if (compileResult != NVRTC_SUCCESS) {
      printCompileLog(*prog);
      error("\ncuda kernel compile failed");
    }

    //  Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(*prog, &ptxSize));

    char * ptx = Calloc(ptxSize, char);
    NVRTC_SAFE_CALL(nvrtcGetPTX(*prog, ptx));

    for(int i = 0; i < kernels.size(); ++i) {
      const char * name;
      NVRTC_SAFE_CALL(nvrtcGetLoweredName(*prog, kernels[i].c_str(), &name));
      (*cudaKernels)[kernels[i]] = new CudaKernel(name, ptx, prog); 
    }
  }
}

extern "C"
void unloadPackage()
{
  std::vector<const char *> ptxs;
  std::vector<nvrtcProgram *> progs;

  std::map<std::string, CudaKernel *>::iterator iter;  
  for (iter = cudaKernels->begin(); iter != cudaKernels->end(); ++iter) {
    ptxs.push_back(iter->second->ptx);
    progs.push_back(iter->second->prog);
    delete iter->second;
  }
  
  delete cudaKernels;

  std::sort(ptxs.begin(), ptxs.end());
  std::unique(ptxs.begin(), ptxs.end());
  std::vector<const char *>::iterator ptx_i;
  for (ptx_i = ptxs.begin(); ptx_i != ptxs.end(); ++ptx_i) {
    Free(*ptx_i);
    }

  std::sort(progs.begin(), progs.end());
  std::unique(progs.begin(), progs.end());
  std::vector<nvrtcProgram *>::iterator prog_i;
  for (prog_i = progs.begin(); prog_i != progs.end(); ++prog_i) {
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(*prog_i));
  }
}

void cudaLaunch(std::string kernelName,
                void * args[],
                const dim3 &gridDim, const dim3 &blockDim,
                cudaStream_t stream)
{
  const CudaKernel * cudaKernel = (*cudaKernels)[kernelName];

  CUmodule module;
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, cudaKernel->ptx, 0, 0, 0));

  CUfunction kernel;
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, cudaKernel->name));

  CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
      gridDim.x, gridDim.y, gridDim.z,    // grid dim
      blockDim.x, blockDim.y, blockDim.z, // block dim
      0, stream,                    // shared mem and stream
      args, 0));                  // arguments
  //  CUDA_SAFE_CALL(cuCtxSynchronize());
  CUDA_SAFE_CALL(cuModuleUnload(module));
}
