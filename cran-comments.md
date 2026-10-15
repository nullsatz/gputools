## Resubmission of an archived package

gputools was archived on 2017-12-19 because check problems were not
corrected. Version 2.0.0 is a rewrite of the build and much of the code:

* The package now builds and installs on every platform, with or without
  CUDA. `configure` looks for a CUDA toolkit (11.7 or later); when there is
  none (as on CRAN's machines, and always on Windows) it compiles a small
  fallback, and each GPU function then stops with an explanation of how to
  get a CUDA build. `gpuAvailable()` returns `FALSE`.
* No `nvcc` is needed at build time: the GPU kernels ship as source in
  `inst/cuda` and are compiled at run time by NVRTC. The CUDA driver
  library is not linked; its entry points are looked up at run time.
* Examples that need a GPU are wrapped in `if (gpuAvailable())`, and tests
  that need one skip without it, so `R CMD check` on a machine without a
  GPU runs the argument checks and the fallback only.

## Test environments

* macOS (aarch64), R 4.6.1, no CUDA: `R CMD check --as-cran`
* Rocky Linux 8, R 4.5.3, CUDA 12.6, NVIDIA RTX 2080 Ti: `R CMD check
  --as-cran`, with the full test suite running on the GPU
* GitHub Actions: Ubuntu with CUDA 11.8, 12.6 and 13.0 (build and install),
  Ubuntu without CUDA, and Windows without CUDA
* win-builder (R-devel)

## R CMD check results

0 errors | 0 warnings | 1 note

* "Package was archived on CRAN": this is the resubmission described above.

## Use of AI tools

The 2.0.0 modernisation (build system, NVRTC kernels, tests, and the new
functions) was done by the maintainer with the help of an AI coding
assistant (Anthropic's Claude). Every function was checked against the base
R function or published method it implements, by tests that compare the
results, and the maintainer reviewed the changes and takes responsibility
for them.
