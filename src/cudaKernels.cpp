#include <map>
#include <string>

#include "cudaKernels.h"

// kernel function name -> ptx code
// load the ptx into a module
// then ask for the CUfunction by name
std::map<std::string, const char *> * cudaKernels;

