# gputools

[![R-CMD-check](https://github.com/nullsatz/gputools/actions/workflows/R-CMD-check.yml/badge.svg)](https://github.com/nullsatz/gputools/actions/workflows/R-CMD-check.yml)
[![r-universe](https://nullsatz.r-universe.dev/badges/gputools)](https://nullsatz.r-universe.dev/gputools)

GPU drop-ins for base-R statistics on wide data. gputools computes the
statistics that cost the most when there are many variables -- every
pairwise correlation, distance or association test, and the clustering
built on them -- on an NVIDIA GPU, and returns the numbers base R would: every
function is tested against its base-R counterpart, to float accuracy in
single precision and to 1e-8 or better in double.

It was written for, and mostly used in, genomics: correlating thousands of
genes or regulatory regions across samples, screening pairs of variables,
clustering expression profiles. The GPU pays off most there; measured
against base R on 8 CPU cores, Kendall's tau ran about 2,900 times faster,
Pearson correlation 16 times, distance matrices 7 times.

## What it computes

**Pairwise statistics over many variables** -- where the GPU wins:

| gputools | base R |
|---|---|
| `gpuCor` (Pearson, Kendall), `gpuTtest` | `cor`, the correlation t-test |
| `gpuDist` | `dist` |
| `gpuHclust`, `gpuDistClust` | `hclust`, `hclust(dist())` |
| `gpuGranger` | Granger causality F-tests for every pair of series |
| `gpuMi` | B-spline mutual information (Daub et al. 2004) |

**Models and linear algebra**, with the same interface as base R:

| gputools | base R |
|---|---|
| `gpuLm`, `gpuLsfit`, `gpuGlm` | `lm`, `lsfit`, `glm` |
| `gpuQr`, `gpuSolve` | `qr`, `solve` |
| `gpuMatMult`, `gpuCrossprod`, `gpuTcrossprod` | `%*%`, `crossprod`, `tcrossprod` |

A multi-threaded BLAS is about as fast as the GPU at matrix products and
small regressions; for general GPU array work in R,
[torch](https://torch.mlverse.org/) is the better tool. The
[benchmarks vignette](vignettes/benchmarks.Rmd) has the measurements.

## Installing

You need an NVIDIA CUDA toolkit, 11.7 or later, to build the package, and
an NVIDIA GPU and driver to run the GPU functions. Install from
[r-universe](https://nullsatz.r-universe.dev):

```r
install.packages("gputools",
  repos = c("https://nullsatz.r-universe.dev", "https://cloud.r-project.org"))
```

or from GitHub:

```r
# install.packages("remotes")
remotes::install_github("nullsatz/gputools")
```

Either way, on Linux gputools is compiled on your machine against your
CUDA toolkit. The prebuilt packages r-universe offers (Windows, macOS, and
its opt-in Linux binaries) are built without CUDA: they install and load,
but their GPU functions cannot run.

### Windows

gputools runs on Windows under **WSL2** (the Windows Subsystem for
Linux), which supports CUDA:

1. Install the ordinary NVIDIA driver for Windows; it also serves WSL2.
   Do not install a separate Linux driver inside WSL.
2. In a WSL2 Linux distribution (Ubuntu, say), install R and the CUDA
   toolkit for WSL, following NVIDIA's
   [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).
3. Install gputools inside WSL as above.

The native Windows package installs and loads, but its GPU functions
cannot run.

### Finding CUDA

`configure` finds CUDA through `CUDA_HOME`, `CUDA_PATH`, `nvcc` on the
`PATH`, or `/usr/local/cuda`; in a conda environment set
`CUDA_HOME=$CONDA_PREFIX`. See [INSTALL](INSTALL) for the details and the
conda packages.

The package also installs and loads on a machine with no GPU (a login
node, CI); `gpuAvailable()` says whether the GPU functions can run there.

## Using it

```r
library(gputools)

# expression of 2,000 genes (columns) in 300 samples (rows)
expr <- matrix(rnorm(300 * 2000), 300, 2000)

r <- gpuCor(expr)                       # all 2,000 x 2,000 correlations
r$coefficients[1:5, 1:5]                # as cor(expr)
r$ts[1:5, 1:5]                          # their t-statistics

tau <- gpuCor(expr, method = "kendall")$coefficients   # as cor(..., "kendall")

tree <- gpuDistClust(t(expr), clustmethod = "average") # hclust(dist(t(expr)))
groups <- cutree(tree, k = 10)

fit <- gpuLm(mpg ~ ., mtcars)           # an "lm": summary(), predict(), ...
```

Each function computes in single precision by default, which is many times
faster than double on most consumer GPUs. Ask for double per call, or for
the whole session:

```r
gpuCor(x, precision = "double")
options(gputools.precision = "double")
```

(`gpuLm`, `gpuLsfit` and `gpuGlm` take `useSingle = FALSE` instead.)

Kernels are compiled for the GPU in use the first time each is needed, so
the first call of a function takes a fraction of a second longer.

Whether the GPU is faster depends on the function, the data size and the
hardware: `vignette("benchmarks", package = "gputools")` has measured
timings against base R.

## Testing on a GPU

`R CMD check` runs the full test suite, but GitHub's runners have no GPU,
so there the GPU tests skip. To run them on a Slurm cluster:

```sh
sbatch -p <gpu partition> ci/slurm-gpu-test.sh <conda env> [<conda env> ...]
```

## Citing

Buckner J, Wilson J, Seligman M, Athey B, Watson S, Meng F (2010). The
gputools package enables GPU computing in R. *Bioinformatics* 26(1),
134-135. [doi:10.1093/bioinformatics/btp608](https://doi.org/10.1093/bioinformatics/btp608)

## License

GPL-3
