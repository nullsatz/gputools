# The routine objects that useDynLib(.registration = TRUE, .fixes = "C_")
# creates when the compiled code loads. CRAN checks a package whose
# SystemRequirements it cannot meet (CUDA) with a fake install, which
# compiles nothing, so these would be reported as undefined globals there.
# test-arguments.R checks this list against the registered routines.
.gpuRoutines <- c(
    "C_gpuAssoc",
    "C_gpuCorPairs",
    "C_gpuCorPerm",
    "C_gpuCorProduct",
    "C_gpuDist",
    "C_gpuDistClust",
    "C_gpuGetDevice",
    "C_gpuGranger",
    "C_gpuGrangerXY",
    "C_gpuHclust",
    "C_gpuKendall",
    "C_gpuKmeans",
    "C_gpuLSFit",
    "C_gpuMatMult",
    "C_gpuMutualInfo",
    "C_gpuPmcc",
    "C_gpuQrDecomp",
    "C_gpuRank",
    "C_gpuRowTtest",
    "C_gpuSetDevice",
    "C_gpuSolve",
    "C_gpuSpearman",
    "C_gpuSvd",
    "C_gpuTtest",
    "C_gputoolsBuildInfo",
    "C_gputoolsSetKernelDir",
    "C_gputoolsUnloadKernels"
)

globalVariables(.gpuRoutines)
