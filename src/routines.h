#ifndef _ROUTINES_H_
#define _ROUTINES_H_

/* The GPU entry points R calls, with their number of arguments. Both
   builds register exactly these: rinterface.cpp with CUDA, nocuda.cpp
   without. */
#define GPUTOOLS_GPU_ROUTINES(X) \
  X(gpuMatMult, 2)               \
  X(gpuPmcc, 7)                  \
  X(gpuCorProduct, 3)            \
  X(gpuCorPairs, 9)              \
  X(gpuAssoc, 9)                 \
  X(gpuKmeans, 4)                \
  X(gpuCorPerm, 5)               \
  X(gpuRowTtest, 6)              \
  X(gpuRank, 2)                  \
  X(gpuSetDevice, 1)             \
  X(gpuGetDevice, 0)             \
  X(gpuTtest, 3)                 \
  X(gpuKendall, 6)               \
  X(gpuSpearman, 6)              \
  X(gpuGranger, 5)               \
  X(gpuGrangerXY, 7)             \
  X(gpuDist, 6)                  \
  X(gpuHclust, 5)                \
  X(gpuDistClust, 7)             \
  X(gpuQrDecomp, 3)              \
  X(gpuSolve, 3)                 \
  X(gpuSvd, 4)                   \
  X(gpuMutualInfo, 8)            \
  X(gpuLSFit, 10)

/* and the housekeeping ones .onLoad, .onUnload and .onAttach use */
#define GPUTOOLS_OTHER_ROUTINES(X) \
  X(gputoolsSetKernelDir, 1)       \
  X(gputoolsUnloadKernels, 0)      \
  X(gputoolsBuildInfo, 0)

#endif /* _ROUTINES_H_ */
