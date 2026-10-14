#ifndef _CUDAUTILS_H_
#define _CUDAUTILS_H_

#include <string>

#include "R.h"
#include "nvrtc.h"
#include "cuda_runtime_api.h"

#define NVRTC_SAFE_CALL(x)                                  \
  do {                                                      \
    nvrtcResult result = x;                                 \
    if (result != NVRTC_SUCCESS)                            \
      error("gputools: %s failed: %s", #x,                  \
            nvrtcGetErrorString(result));                   \
  } while(0)

// Raises an R error unless a CUDA device can be used.
void requireDevice();

// Which build of a kernel to run: the kernels are written in terms of
// "real", float or double.
enum class Precision { Single, Double };

template <typename T> constexpr Precision precisionOf();
template <> constexpr Precision precisionOf<float>() { return Precision::Single; }
template <> constexpr Precision precisionOf<double>() { return Precision::Double; }

// Launches a kernel from inst/cuda, compiling its file first if needed.
void cudaLaunch(std::string kernelName,
                void * args[],
                const dim3 &gridDim, const dim3 &blockDim,
                Precision precision = Precision::Single,
                cudaStream_t stream = NULL);

#endif /* _CUDAUTILS_H_ */
