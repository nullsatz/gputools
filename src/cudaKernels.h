#ifndef _CUDA_KERNELS_H_
#define  _CUDA_KERNELS_H_

#include <map>
#include <string>

// kernel function name -> ptx code
// load the ptx into a module
// then ask for the CUfunction by name
extern std::map<std::string, const char *> * cudaKernels;

#endif /* _CUDA_KERNELS_H_ */
