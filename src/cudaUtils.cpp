#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <tuple>

#include "R.h"
#include "Rinternals.h"
#include "nvrtc.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "cudaUtils.h"
#include "gpublas.h"
#include "gpusolver.h"

// Kernels are compiled from inst/cuda/<file>.cu with NVRTC the first time
// one of them is launched, for the architecture of the current device, and
// the loaded module is kept for the rest of the session.
//
// The few driver API functions needed are looked up through the runtime
// rather than linked, so the package does not depend on libcuda and still
// loads on a machine without the NVIDIA driver.

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
        "gpuSDNoTest", "gpuPMCCNoTest" } },
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
    { "kendall", { "gpuKendall", "gpuSpearmanPairwise" } },
    { "rank", { "gpuRankColumns" } },
    { "pairs",
      { "gpuThresholdPairs", "gpuTopPerColumn", "gpuColumnNorms",
        "gpuScaleColumns" } },
    { "mi", { "scale", "get_bin_scores", "get_entropy", "get_mi" } },
    { "qrdecomp",
      { "getColNorms" } },
  };
  return files;
}

std::string kernelDir;

// Each file is compiled once per precision: kernels are written in terms
// of "real", which this prelude defines. #line keeps NVRTC's error line
// numbers those of the .cu file.
std::string prelude(bool isDouble)
{
  return std::string("typedef ") + (isDouble ? "double" : "float") +
    " real;\n"
    "#define REAL_INF ((real) __int_as_float(0x7f800000U))\n"
    "#define REAL_NAN ((real) __int_as_float(0x7fffffffU))\n"
    "#line 1\n";
}

// A module per (device, file, precision), and the kernels found in it.
typedef std::tuple<int, std::string, bool> ModuleKey;
std::map<ModuleKey, CUmodule> modules;
std::map<ModuleKey, CUfunction> kernels;

template <typename F>
F driverFunction(const char * name)
{
  void * fn = NULL;
#if CUDART_VERSION >= 12050
  cudaDriverEntryPointQueryResult found;
  CUDART_CALL(cudaGetDriverEntryPointByVersion(name, &fn, CUDART_VERSION,
                                               cudaEnableDefault, &found));
#else
  CUDART_CALL(cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault));
#endif
  if (fn == NULL)
    error("gputools: the CUDA driver has no %s", name);
  return (F) fn;
}

struct Driver {
  decltype(&cuGetErrorName) getErrorName;
  decltype(&cuModuleLoadData) moduleLoadData;
  decltype(&cuModuleGetFunction) moduleGetFunction;
  decltype(&cuModuleUnload) moduleUnload;
  decltype(&cuLaunchKernel) launchKernel;
};

const Driver & driver()
{
  static Driver d;
  static bool found = false;
  if (!found) {
    d.getErrorName = driverFunction<decltype(d.getErrorName)>("cuGetErrorName");
    d.moduleLoadData =
      driverFunction<decltype(d.moduleLoadData)>("cuModuleLoadData");
    d.moduleGetFunction =
      driverFunction<decltype(d.moduleGetFunction)>("cuModuleGetFunction");
    d.moduleUnload = driverFunction<decltype(d.moduleUnload)>("cuModuleUnload");
    d.launchKernel = driverFunction<decltype(d.launchKernel)>("cuLaunchKernel");
    found = true;
  }
  return d;
}

void driverCheck(CUresult status, const char * call)
{
  if (status != CUDA_SUCCESS) {
    const char * name = "unknown error";
    driver().getErrorName(status, &name);
    error("gputools: %s failed: %s", call, name);
  }
}

#define DRIVER_CALL(x) driverCheck((x), #x)

std::map<int, cublasHandle_t> blasHandles;
std::map<int, cusolverDnHandle_t> solverHandles;

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

Compiled compile(const KernelFile & kf, int device, bool isDouble)
{
  std::string src = prelude(isDouble) + readSource(kf.file);

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
    error("gputools: compiling %s.cu failed (%s, %s):\n%s", kf.file,
          arch.c_str(), isDouble ? "double" : "single", log.c_str());
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

void loadFile(const KernelFile & kf, int device, bool isDouble)
{
  Compiled c = compile(kf, device, isDouble);

  CUmodule module;
  DRIVER_CALL(driver().moduleLoadData(&module, c.ptx.c_str()));
  modules[ModuleKey(device, kf.file, isDouble)] = module;

  for (size_t i = 0; i < kf.kernels.size(); ++i) {
    CUfunction kernel;
    DRIVER_CALL(driver().moduleGetFunction(&kernel, module,
                                           c.lowered[i].c_str()));
    kernels[ModuleKey(device, kf.kernels[i], isDouble)] = kernel;
  }
}

} // namespace

cublasHandle_t gpuBlasHandle()
{
  int device = 0;
  CUDART_CALL(cudaGetDevice(&device));
  auto h = blasHandles.find(device);
  if (h != blasHandles.end())
    return h->second;
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  blasHandles[device] = handle;
  return handle;
}

cusolverDnHandle_t gpuSolverHandle()
{
  int device = 0;
  CUDART_CALL(cudaGetDevice(&device));
  auto h = solverHandles.find(device);
  if (h != solverHandles.end())
    return h->second;
  cusolverDnHandle_t handle;
  CUSOLVER_CALL(cusolverDnCreate(&handle));
  solverHandles[device] = handle;
  return handle;
}

void gpuSolverCheck(cusolverStatus_t status, const char * call)
{
  if (status != CUSOLVER_STATUS_SUCCESS)
    error("gputools: %s failed: cuSOLVER status %d", call, (int) status);
}

void gpuBlasCheck(cublasStatus_t status, const char * call)
{
  if (status != CUBLAS_STATUS_SUCCESS)
    error("gputools: %s failed: %s", call, cublasGetStatusString(status));
}

void gpuCudaCheck(cudaError_t status, const char * call)
{
  if (status != cudaSuccess)
    error("gputools: %s failed: %s", call, cudaGetErrorString(status));
}

void requireDevice()
{
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    cudaGetLastError();  // clear it, so later checks do not report it again
    error("gputools: no usable CUDA device: %s", cudaGetErrorString(status));
  }
  if (count == 0)
    error("gputools: no CUDA device found");
}

void cudaLaunch(std::string kernelName,
                void * args[],
                const dim3 &gridDim, const dim3 &blockDim,
                Precision precision, cudaStream_t stream)
{
  bool isDouble = precision == Precision::Double;
  int device = 0;
  CUDART_CALL(cudaGetDevice(&device));
  // makes the device's primary context current for the driver API
  CUDART_CALL(cudaFree(0));

  ModuleKey key(device, kernelName, isDouble);
  auto k = kernels.find(key);
  if (k == kernels.end()) {
    loadFile(fileForKernel(kernelName), device, isDouble);
    k = kernels.find(key);
  }

  DRIVER_CALL(driver().launchKernel(k->second,
                                    gridDim.x, gridDim.y, gridDim.z,
                                    blockDim.x, blockDim.y, blockDim.z,
                                    0, stream, args, NULL));
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
    cudaSetDevice(std::get<0>(m.first));
    driver().moduleUnload(m.second);
  }
  for (auto & h : blasHandles) {
    cudaSetDevice(h.first);
    cublasDestroy(h.second);
  }
  for (auto & h : solverHandles) {
    cudaSetDevice(h.first);
    cusolverDnDestroy(h.second);
  }
  cudaSetDevice(current);
  modules.clear();
  kernels.clear();
  blasHandles.clear();
  solverHandles.clear();
  return R_NilValue;
}

}
