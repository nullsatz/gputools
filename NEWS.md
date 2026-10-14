# gputools 2.0.0

## Wrong results fixed

These gave wrong answers in earlier versions:

- `gpuDist(method = "canberra")` used |a - b| / |a + b|; it now uses R's
  |a - b| / (|a| + |b|), leaving out 0/0 terms as `dist()` does.
- `gpuHclust(method = "ward")` swapped the two cluster sizes in its update,
  so merges of unequal-sized clusters got the wrong height.
- `gpuGranger(x, y)` with a single column in `x` tested every `y` column
  but the last against the wrong series.
- `gpuQr` chose its pivots from wrongly updated column norms, so it could
  report too low a rank.
- `gpuGranger` p-values below about 1e-7 were rounded to 0.
- The `gpuGranger` documentation described `fStatistics` the other way
  round from what the code computes; it now says what the code does:
  `fStatistics[i, j]` tests whether `x[, i]` Granger-causes `x[, j]` (or
  `y[, j]`).

And these crashed or failed:

- `qr.X()` and other base-R functions given a `gpuQr()` result could
  corrupt memory and crash R; `gpuTtest`, `gpuSolve` and `gpuHclust`
  results carried the same hazard.
- `gpuTtest` could not be called at all, and `gpuGlm` stopped with
  "argument 16 is empty" (#23).
- `gpuMatMult` read integer matrices as doubles.
- `gpuDistClust` leaked the GPU memory of its distance matrix.

## New

- Every GPU function can compute in double precision:
  `precision = "double"`, or `options(gputools.precision = "double")`
  for the session. `useSingle = FALSE` in `gpuLm`, `gpuLsfit` and
  `gpuGlm`, which used to stop with "not yet implemented", now works.
- `gpuAvailable()` says whether a usable CUDA device is present.
- A benchmarks vignette with measured timings against base R.

## Builds and runs on current systems

- Builds with CUDA 11.7 through 13 and R 4.0 or later (tested with R 4.5);
  the package had not compiled since R and CUDA moved on (#21, #26 and
  others).
- Installs and loads without a GPU or the NVIDIA driver; the GPU functions
  then stop with "no usable CUDA device".
- `configure` finds CUDA through `CUDA_HOME`, `CUDA_PATH`, `nvcc` on the
  `PATH`, conda environments or `/usr/local/cuda`.
- Kernels are compiled for the GPU in use the first time they are needed
  and kept loaded: loading the package takes milliseconds, and
  `gpuHclust` on 3000 points went from 26 s to 0.3 s.
- Uses the current cuBLAS API, `.Call` and registered routines; about
  1500 lines of unused code are gone.
- A test suite compares every function with base R, and continuous
  integration checks the package against several CUDA versions.

# gputools 1.1
- this is a point release intended to trial cuda run-time compilation (nvrtc)
  it is only used for the kendall correlation option of gpuCor
  but nvrtc will be used for all cuda kernel code in future releases

# gputools 0.5
- switched back to autotools options

# gputools 0.28
- updated installation procedure
- updated OSX compatibility
- removed CULA dependencies

# gputools 0.27
- removed LICENSE file and updated DESCRIPTION file
- fixed R documentation files
- removed nonfunctional svm code

# gputools 0.26
- fixes to several potential bugs:  
	There was a variable in the QR code which was never  
	deallocated.  Also, there were several variables freed with  
	the wrong method. Also, added a missing cublasInit/Shutdown pair.
	Thanks to Eunjin Lee's error reporting!

# gputools 0.24
- added support for configure argument --with-compiler-bindir and
	environment variable COMPILER_BINDIR to support gcc located somewhere
	other than the user's path. Use one of these to get the package
	installed in flavors of Linux that ship with default gcc version > 4.4.

# gputools 0.24
- A couple of changes to the Makefile:
	* added -I. for compatibility with R 2.12
	* switched from CC to NVCC for compat with ~/.R/Makevars

# gputools 0.21
- fixed another bug in gpuMi

# gputools 0.2
- added some linear and nonlinear regression functions

- added some configure args --with-r-lib and --with-r-include (see INSTALL
notes section)

- fixed a bug in gpuMi

- added a demo of CULA use: fastICA with CULA's svd plugged in.

# gputools 0.1.3
- Mac os x and nvidia cards with compute capability less than 1.3 are now
supported. Systems lacking a gpu altogether also receive some support. See the
file INSTALL for details.

- gpuHclust, gpuDist, and gpuQr now output the same classes as R's base hclust,
dist, and qr functions.

- The c code now uses r's api. Host memory allocation and error reporting
have received a facelift.

- gpuGranger now reports the correct p-values and is slightly faster for small
data sets.

- Some bugs with gpuCor's pearson method have been fixed.
