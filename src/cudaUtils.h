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

void cudaLaunch(std::string kernelName,
                void * args[],
                const dim3 &gridDim, const dim3 &blockDim,
                cudaStream_t stream = NULL);

#endif /* _CUDAUTILS_H_ */
