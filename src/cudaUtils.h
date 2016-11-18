#ifndef _CUDAUTILS_H_
#define _CUDAUTILS_H_

#include <string>

#include "R.h"
#include "nvrtc.h"
#include "cuda_runtime_api.h"

#define NVRTC_SAFE_CALL(x)                           \
  do {                                               \
    nvrtcResult result = x;                          \
    if (result != NVRTC_SUCCESS) {                   \
      error("\nerror: %d failed with error %s\n", x, \
            nvrtcGetErrorString(result));            \
    }                                                \
  } while(0)

#define CUDA_SAFE_CALL(x)                                       \
  do {                                                          \
    CUresult result = x;                                        \
    if (result != CUDA_SUCCESS) {                               \
      const char *msg;                                          \
      cuGetErrorName(result, &msg);                             \
      error("\nerror: %d failed with error %s\n", x, msg);      \
    }                                                           \
  } while(0)

void cudaLaunch(std::string kernelName,
                void * args[],
                const dim3 &gridDim, const dim3 &blockDim,
                cudaStream_t stream = NULL);

#endif /* _CUDAUTILS_H_ */
