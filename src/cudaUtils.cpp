#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "R.h"
#include "Rinternals.h"
#include "nvrtc.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "cudaUtils.h"

// Kernels are compiled from inst/cuda/<file>.cu with NVRTC the first time
// one of them is launched, for the architecture of the current device, and
// the loaded module is kept for the rest of the session.

namespace {

struct KernelFile {
  const char * file;
  std::vector<std::string> kernels;
};

const std::vector<KernelFile> & kernelFiles()
{
  static const std::vector<KernelFile> files = {
    { "correlation",
      { "gpuSignif", "gpuMeans", "gpuSD", "gpuPMCC", "gpuMeansNoTest",
        "gpuSDNoTest", "gpuPMCCNoTest", "dUpdateSignif", "noNAsPmccMeans" } },
    { "distance",
      { "euclidean_kernel_same", "maximum_kernel_same",
        "manhattan_kernel_same", "canberra_kernel_same",
        "binary_kernel_same", "minkowski_kernel_same" } },
    { "granger", { "getRestricted", "getUnrestricted", "ftest" } },
    { "hcluster",
      { "complete_kernel", "wpgma_kernel", "average_kernel", "median_kernel",
        "centroid_kernel", "flexible_group_kernel", "flexible_kernel",
        "ward_kernel", "mcquitty_kernel", "single_kernel", "convert_kernel",
        "find_min1_kernel", "find_min2_kernel" } },
    { "kendall", { "gpuKendall" } },
    { "mi", { "scale", "get_bin_scores", "get_entropy", "get_mi" } },
    { "qrdecomp",
      { "getColNorms", "gpuFindMax", "gpuSwapCol", "makeHVector" } },
  };
  return files;
}

std::string kernelDir;

// A module per (device, file), and the functions found in it.
std::map<std::pair<int, std::string>, CUmodule> modules;
std::map<std::pair<int, std::string>, CUfunction> functions;

const KernelFile & fileForKernel(const std::string & kernelName)
{
  for (const KernelFile & kf : kernelFiles())
    if (std::find(kf.kernels.begin(), kf.kernels.end(), kernelName)
        != kf.kernels.end())
      return kf;
  error("gputools: no kernel named '%s'", kernelName.c_str());
}

std::string readSource(const std::string & file)
{
  std::string path = kernelDir + "/" + file + ".cu";
  std::ifstream in(path.c_str());
  if (!in)
    error("gputools: cannot read kernel source %s", path.c_str());
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

// The newest virtual architecture both this NVRTC and the device support,
// so the driver JIT-compiles the PTX for the device.
std::string archOption(int device)
{
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  int want = 10 * major + minor;

  int numArchs = 0;
  NVRTC_SAFE_CALL(nvrtcGetNumSupportedArchs(&numArchs));
  std::vector<int> archs(numArchs);
  NVRTC_SAFE_CALL(nvrtcGetSupportedArchs(archs.data()));

  int best = 0;
  for (int a : archs)
    if (a <= want && a > best) best = a;
  if (best == 0)
    error("gputools: device compute capability %d.%d is older than any "
          "this CUDA toolkit supports", major, minor);
  return "--gpu-architecture=compute_" + std::to_string(best);
}

struct Compiled {
  std::string ptx;
  std::vector<std::string> lowered;  // mangled names, parallel to kernels
};

Compiled compile(const KernelFile & kf, int device)
{
  std::string src = readSource(kf.file);

  nvrtcProgram prog;
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, src.c_str(), kf.file,
                                     0, NULL, NULL));
  for (const std::string & k : kf.kernels)
    NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, k.c_str()));

  std::string arch = archOption(device);
  const char * options[] = { "--use_fast_math", arch.c_str() };
  nvrtcResult compiled = nvrtcCompileProgram(prog, 2, options);

  if (compiled != NVRTC_SUCCESS) {
    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    nvrtcGetProgramLog(prog, &log[0]);
    nvrtcDestroyProgram(&prog);
    error("gputools: compiling %s.cu failed (%s):\n%s", kf.file,
          arch.c_str(), log.c_str());
  }

  Compiled out;
  size_t ptxSize = 0;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  out.ptx.resize(ptxSize);  // includes the terminating NUL
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, &out.ptx[0]));

  // lowered names belong to prog, so copy them before destroying it
  for (const std::string & k : kf.kernels) {
    const char * name;
    NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, k.c_str(), &name));
    out.lowered.push_back(name);
  }
  nvrtcDestroyProgram(&prog);
  return out;
}

void loadFile(const KernelFile & kf, int device)
{
  Compiled c = compile(kf, device);

  CUmodule module;
  CUDA_SAFE_CALL(cuModuleLoadData(&module, c.ptx.c_str()));
  modules[{device, kf.file}] = module;

  for (size_t i = 0; i < kf.kernels.size(); ++i) {
    CUfunction fn;
    CUDA_SAFE_CALL(cuModuleGetFunction(&fn, module, c.lowered[i].c_str()));
    functions[{device, kf.kernels[i]}] = fn;
  }
}

} // namespace

void cudaLaunch(std::string kernelName,
                void * args[],
                const dim3 &gridDim, const dim3 &blockDim,
                cudaStream_t stream)
{
  int device = 0;
  CUDA_SAFE_CALL(cuInit(0));
  // makes the device's primary context current for the driver API calls
  cudaFree(0);
  cudaGetDevice(&device);

  auto fn = functions.find({device, kernelName});
  if (fn == functions.end()) {
    loadFile(fileForKernel(kernelName), device);
    fn = functions.find({device, kernelName});
  }

  CUDA_SAFE_CALL(
    cuLaunchKernel(fn->second,
      gridDim.x, gridDim.y, gridDim.z,
      blockDim.x, blockDim.y, blockDim.z,
      0, stream, args, 0));
}

extern "C" {

SEXP gputoolsSetKernelDir(SEXP dir)
{
  kernelDir = CHAR(STRING_ELT(dir, 0));
  return R_NilValue;
}

SEXP gputoolsUnloadKernels()
{
  int current = 0;
  cudaGetDevice(&current);
  for (auto & m : modules) {
    cudaSetDevice(m.first.first);
    cuModuleUnload(m.second);
  }
  cudaSetDevice(current);
  modules.clear();
  functions.clear();
  return R_NilValue;
}

}
