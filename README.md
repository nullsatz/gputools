# gputools

[![R-CMD-check](https://github.com/nullsatz/gputools/actions/workflows/R-CMD-check.yml/badge.svg)](https://github.com/nullsatz/gputools/actions/workflows/R-CMD-check.yml)
[![r-universe](https://nullsatz.r-universe.dev/badges/gputools)](https://nullsatz.r-universe.dev/gputools)

GPU versions of base R's statistics for wide data, on NVIDIA GPUs.

Wide data has many more variables than samples: thousands of genes in a
few hundred samples, say. The statistics that cost the most on such data
are the ones about every pair of variables (correlation, distance,
association tests) and the clustering built on them. gputools computes
those on the GPU and returns the numbers base R would. Every function is
tested against its base R counterpart, to float accuracy in single
precision and to 1e-8 or better in double.

It was written for genomics, and that is where it is mostly used:
correlating thousands of genes or regulatory regions across samples,
screening pairs of variables, clustering expression profiles. Against base
R on 8 CPU cores, Kendall's tau ran about 2,900 times faster, Pearson
correlation 16 times, and distance matrices 7 times.

## What it computes

Pairwise statistics over many variables, where the GPU wins:

| gputools | base R |
|---|---|
| `gpuCor` (Pearson, Kendall, Spearman; with p-values) | `cor`, `cor.test` |
| `gpuCorPairs` | `cor` then `which(abs(r) >= t)`, without the matrix |
| `gpuAssoc` | `lm(y[, j] ~ x[, i] + covariates)` for every pair: an eQTL scan, as Matrix eQTL |
| `gpuCorPerm` | permutation and maxT p-values for every correlation |
| `gpuRowTtest` | `t.test` of every row, with permutation and maxT p-values |
| `gpuTtest` | the correlation t-test |
| `gpuDist` | `dist` |
| `gpuHclust`, `gpuDistClust` | `hclust`, `hclust(dist())` |
| `gpuKmeans` | `kmeans(algorithm = "Lloyd")` |
| `gpuGranger` | Granger causality F-tests for every pair of series |
| `gpuMi` | B-spline mutual information (Daub et al. 2004) |
| `gpuPrcomp`, `gpuSvd` | `prcomp`, `svd` |

Models and linear algebra, with the same interface as base R:

| gputools | base R |
|---|---|
| `gpuLm`, `gpuLsfit`, `gpuGlm` | `lm`, `lsfit`, `glm` |
| `gpuQr`, `gpuSolve` | `qr`, `solve` |
| `gpuMatMult`, `gpuCrossprod`, `gpuTcrossprod` | `%*%`, `crossprod`, `tcrossprod` |

The second group is there for convenience. R's own `lm`, `glm` and `%*%`
were as fast or faster at every size we measured, and for general GPU
array work in R, [torch](https://torch.mlverse.org/) is the better tool.

## Installing

To build the package you need an NVIDIA CUDA toolkit, 11.7 or later; to
run the GPU functions, an NVIDIA GPU and driver. Install from
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

The package also installs and loads on a machine with no GPU, such as a
login node or a CI runner; `gpuAvailable()` says whether the GPU
functions can run there.

### Finding CUDA

`configure` finds CUDA through `CUDA_HOME`, `CUDA_PATH`, `nvcc` on the
`PATH`, or `/usr/local/cuda`. In a conda environment, set
`CUDA_HOME=$CONDA_PREFIX`. See [INSTALL](INSTALL) for the details and the
conda packages.

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

## How fast

Whether the GPU is faster depends on the function, the data size and the
hardware. The numbers here are from an RTX A5500 in single precision,
against base R on 8 CPU cores.

**Where the GPU starts to win.** Below these sizes, base R is faster:

| function | faster than base R from |
|---|---|
| `gpuCor`, Kendall | 5 variables |
| `gpuCor`, Spearman | 10 variables |
| `gpuCor`, Pearson | 30 variables |
| `gpuGranger` | 5 series |
| `gpuDist` | 100 points |
| `gpuPrcomp` | 100 variables |
| `gpuDistClust` | 300 points |
| `gpuSolve` | 300 x 300 |
| `gpuQr` | 1,000 columns |

**At genomics sizes**, the gains are large:

| job | gputools | base R or the standard tool |
|---|---|---|
| every pair with \|r\| >= 0.8 among 10,000 variables (`gpuCorPairs`) | 0.02 s | 6.5 s (`cor()` and `which()`) |
| the same among 50,000 variables | 0.09 s | not run (the matrix is 20 GB) |
| eQTL scan: 20,000 genes, 50,000 SNPs, 500 samples, 5 covariates (`gpuAssoc`) | 0.37 s | 11 s (Matrix eQTL, the same associations) |
| 100,000 points into 1,000 clusters (`gpuKmeans`) | 0.25 s | 14.5 s (`kmeans()`) |
| 1,000 permutations of every correlation among 1,000 x 1,000 variables, maxT (`gpuCorPerm`) | 0.13 s | 20 s |
| 1,000 permutations of t tests of 20,000 genes (`gpuRowTtest`) | 0.06 s | 23 s |

Permutation tests gain the most, because a thousand permutations cost a
thousand times the test in R and one batched computation on the GPU.
`gpuCorPairs` never builds the correlation matrix, so it also runs at sizes
where `cor()` cannot.

Each call carries a fixed cost of about 50 µs, so **don't loop over small
pieces**: put all the variables in one matrix and make one call.
`vignette("benchmarks", package = "gputools")` has the full measurements
and the script to repeat them on your hardware.

## Testing on a GPU

`R CMD check` runs the full test suite, but GitHub's runners have no GPU,
so there the GPU tests skip. To run them on a Slurm cluster:

```sh
sbatch -p <gpu partition> ci/slurm-gpu-test.sh <conda env> [<conda env> ...]
```

## Used in

- The FANTOM5 atlas of active enhancers across human cell types and tissues
  ([Andersson et al., *Nature* 2014](https://doi.org/10.1038/nature12787))
  clustered enhancers by their expression with gputools' hierarchical
  clustering on an NVIDIA GPU.

Using gputools in published work? Let us know in an
[issue](https://github.com/nullsatz/gputools/issues) and we will list it.

## Citing

Buckner J, Wilson J, Seligman M, Athey B, Watson S, Meng F (2010). The
gputools package enables GPU computing in R. *Bioinformatics* 26(1),
134-135. [doi:10.1093/bioinformatics/btp608](https://doi.org/10.1093/bioinformatics/btp608)

## Acknowledgements

Thanks to Craig Stark at UC Irvine for donating time on his lab's Mac to
the original package.

## License

GPL-3
