# gputools algorithm review (Phase 5)

Each GPU method checked against its published or standard definition: what
the definition is, what the code computes, the verdict, and the test that
now holds it there. References that describe an algorithm the package
implements are checked against that algorithm; the S/R model references in
the `lm`/`glm` help pages define what the models mean, which base R
implements, so those functions are checked against `lm()`/`glm()`.

## Kendall's tau (`gpuCor(method = "kendall")`)

**Definition.** Kendall (1938, 1945). With ties, the statistic in general
use, and the one `cor(method = "kendall")` computes, is tau-b:

    tau_b = (C - D) / sqrt((n0 - Tx) (n0 - Ty)),  n0 = n (n - 1) / 2

where C and D count concordant and discordant pairs and Tx, Ty the pairs
tied in x and in y. A pair tied in either variable is neither concordant
nor discordant.

**Code before.** Counted a pair as concordant if both variables moved the
same way *or were tied in both*, and every other pair -- including those
tied in only one variable -- as discordant, then returned
(C' - D') / n0. Without ties this equals tau-a = tau-b; with ties it is
neither. On ordinal data (4-5 levels, 60 observations) it was off by up to
0.32, with the wrong sign in some cases (0.050 returned as -0.245).
The counts were also accumulated in the floating-point type, so in single
precision they stopped being exact beyond 2^24 pairs (about 5,800
observations), where C - D cancels badly for small tau.

**Now.** Counts C, D, Tx and Ty as 64-bit integers and returns tau-b.
Matches a from-the-definition implementation and `cor()` (which agree to
7e-18). Tests: tied ordinal data in both precisions; 6,000 integer
observations (~18M pairs) matching `cor()` to 1e-12 in both precisions.

## Correlation t-statistic (`gpuTtest`, `gpuCor(...)$ts`)

**Definition.** For a Pearson correlation r from n pairs, the statistic for
H0: rho = 0 is t = r sqrt((n - 2) / (1 - r^2)) on n - 2 degrees of
freedom, as in `cor.test()`. It is +-Inf at |r| = 1 and undefined for
n < 3.

**Code before.** `gpuCor`'s `$ts` replaced t with 10000 whenever
r >= 0.999, whatever n was (r = 0.9995 from 10 pairs gave 10000 instead
of 89.4), and did not do the same for r <= -0.999. In double precision it
also took the square root in single precision (`sqrtf`, left over from the
precision templating). `gpuTtest`'s GPU kernel produced NaN for |r| a
rounding error above 1.

**Now.** One rule on the host and the GPU: r clamped to [-1, 1], +-Inf at
|r| = 1, NaN with fewer than three pairs, otherwise the formula in full
precision. Tests: r near +-1, r = +-1, n = 2, and `gpuCor`'s `$ts` against
the formula, in both precisions.

## Hierarchical clustering (`gpuHclust`, `gpuDistClust`)

**Definition.** Every method is a Lance-Williams (1967) update: when
clusters i and j merge, the dissimilarity to each other cluster k becomes

    d(k, i+j) = ai d(k,i) + aj d(k,j) + b d(i,j) + g |d(k,i) - d(k,j)|

with the published coefficients per method (single, complete, average =
UPGMA, mcquitty = WPGMA, median = WPGMC, centroid = UPGMC, Ward); Lance &
Williams' flexible-beta method, ai = aj = (1 - b) / 2; and Belbin, Faith &
Milligan's (1992) flexible UPGMA ("flexible group"),
ai = (1 - b) ni / (ni + nj). `tests/testthat/helper-gpu.R` implements the
update directly from these coefficients (`lanceWilliams()`); it agrees with
`hclust()` to 4e-15 on the seven methods base R has.

**Code before.**
- `ward` had the two cluster sizes swapped (fixed in 2.0.0).
- `"flexible group"` was never computed: R passed `"flexible group"`, the
  C code looked for `"flexible_group"`, and an unknown name fell back to
  single linkage. Its kernel, which had never run, also had the cluster
  sizes swapped, the same mistake as Ward's.
- `wpgma` and `mcquitty` are the same method under two names, as they
  should be.

**Now.** The C code takes the names R passes and stops on an unknown one
instead of defaulting; the flexible-UPGMA weights are each cluster's own
size. All ten methods match `lanceWilliams()` to 1e-10 in double precision
(and float accuracy in single).

**Open: the flexible methods' parameters.** Both use fixed lambda = beta =
0.5 (that is, ai + aj = 1/2, b = +1/2), with no argument to change them.
The formula is applied correctly for those values, but Lance & Williams
recommend b around -0.25 (slightly space-dilating); b = +0.5 is strongly
space-contracting. Options: expose `beta` (with ai, aj following from it)
and keep 0.5 as the default for compatibility, or change the default to the
literature's -0.25.

## B-spline mutual information (`gpuMi`)

**Definition.** Daub, Steuer, Selbig & Kloska (2004), BMC Bioinformatics
5:118. Each variable is scaled linearly onto the spline domain
[0, M - k + 1]; each value is spread over the M bins with weights
B_{i,k}(z), Cox-de Boor B-splines of order k on the knot vector
0..0, 1, 2, ..., M-k+1..M-k+1; bin probabilities are the mean weights and
joint probabilities the mean products of weights; entropies are in bits and
MI(x, y) = H(x) + H(y) - H(x, y). (The paper's equations are images; the
knot vector and scaling follow from its text and are confirmed by its worked
example.)

**Reference.** `daubMI()` in `tests/testthat/helper-gpu.R`, written from
the paper. It reproduces the paper's worked example exactly (M = 3, k = 2:
the Table 1 weights, p(a) = (1.8, 2.4, 1.8)/6, H(x) =
log2(10) - 0.6 log2(3) - 0.4 log2(4) = 1.571, MI = 0.4489, printed as 0.45).

**Code.** Correct: `gpuMi` matches the reference to 5e-15 in double
precision for bins 5 and 10 and spline orders 1-4, and gives the paper's
0.4489 on its example. Tests: the worked example and the random-data
comparison, in both precisions.

## Granger causality (`gpuGranger`)

**Definition.** Granger (1969): Y Granger-causes X if X is better
predicted from its own past and Y's past than from its own past alone. The
usual test, used here, compares by F-test the regressions

    X_t ~ 1 + X_{t-1..t-p}               (restricted)
    X_t ~ 1 + X_{t-1..t-p} + Y_{t-1..t-p} (unrestricted)

on the n - p usable time points: F with p and n - 3p - 1 degrees of
freedom.

**Code.** The statistic and degrees of freedom match: in double precision
the F statistics agree with this pair of `lm()` fits to 1e-8 and the
p-values to 1e-6 (2.0.0 fixed the x/y form with one x column and p-values
rounded to 0).

**Orientation (deferred here from 2.0.0).** Granger's definition does not
say how to lay a matrix of tests out; that is a convention. The code's is
`fStatistics[i, j]` = "does column i Granger-cause column j", the usual
from-row-to-column reading of a directed graph's adjacency matrix,
consistent between the single-matrix and x/y forms; 2.0.0 corrected the
documentation to it. Kept. The result matrices now carry dimnames
`cause` x `effect` (from the variables' names), so a printed result shows
which way it reads, and the diagonal -- a series against itself, which is no
test -- is `NA` rather than F = 0, p = 1.

## QR decomposition and least squares (`gpuQr`, `gpuLm`, `gpuLsfit`, `gpuGlm`)

**Definitions.** `gpuQr` cites Bjorck (1996): Householder QR with column
pivoting (Businger-Golub), whose rank test is relative,
|r_kk| <= tol |r_11|. `gpuLm`/`gpuLsfit`/`gpuGlm` stand in for `lm`,
`lsfit` and `glm`, which use LINPACK's `dqrdc2`: limited pivoting, where a
column whose norm below the rows already reduced falls under tol times its
own original norm is moved to the end (the rest keep their order); every
column is still reduced, and the rank is the number not moved. Both tests
are scale-invariant. The least-squares code cited Bischof & Van Loan
(1987) for its blocked (WY) Householder QR.

**Code before.**
- *Rank test (both).* Stopped when the remaining norm fell below
  tol (1 + |r_11|): relative for data of order one, but an absolute
  threshold for data in small units. Data in units of 1e-8 got rank 0 from
  `gpuQr` and all-`NA` coefficients from `gpuLm`, where `lm` fits normally;
  with `tol = 1e-4`, already at 1e-5.
- *Collinearity (`gpuLm`, `gpuLsfit`, `gpuGlm`).* The blocked QR measured
  each column's norm over the whole column, which orthogonal reflections
  never change, so a column that is a combination of earlier ones was never
  detected. Exactly collinear predictors gave no `NA` and huge
  coefficients: about 1e3 in single precision, 1.6e13 in double (`lm`:
  0.98, -1.01, `NA`).
- *`gpuGlm` in single precision* passed `glm.fit`'s QR tolerance,
  min(1e-7, epsilon / 1000) = 1e-8 there, below what floats resolve, so it
  could not detect collinearity even with a correct QR.

**Now.**
- `gpuLm`/`gpuLsfit`/`gpuGlm` use a port of `dqrdc2` (unblocked, remaining
  norms recomputed exactly each step, LINPACK's compact output), so they
  alias the same columns as `lm`/`lsfit`/`glm`. This replaces the blocked
  WY QR.
- `gpuQr` keeps full column pivoting, with Bjorck's relative rank test.
- `gpuGlm` uses the float-sized tolerance (1e-4) in single precision and
  `glm.fit`'s own in double.
- Tests, in both precisions: exactly collinear predictors (same `NA` and
  coefficients as `lm`/`glm`/`lsfit`, including `lsfit`'s collinearity
  warning), a zero column, and data in units of 1e-9 and 1e9.

Speed: the `dqrdc2` port costs nothing measurable against the blocked QR it
replaced (RTX 2080 Ti: `gpuLm` on 100,000 x 50, 0.23 s against 0.25 s;
`gpuLsfit` on 4,000 x 500, 0.11 s both).

## Distances (`gpuDist`) and Pearson correlation (`gpuCor`)

**Definitions.** The metrics `dist()` documents (euclidean, maximum,
manhattan, canberra, binary, minkowski) and the Pearson product-moment
correlation, with `use = "pairwise.complete.obs"` meaning each pair of
columns uses the rows complete in both (means and deviations recomputed per
pair), as `cor()` does. Base R is the definition here.

**Code.** Canberra used |a - b| / |a + b| (fixed in 2.0.0: R's
|a - b| / (|a| + |b|), 0/0 terms left out). Otherwise correct: every metric
and both `use` modes match `dist()`/`cor()` to 1e-10 in double precision.

## Kernel arguments

Kernels take their arguments as an array of pointers, so nothing checks
that the host passes the type a kernel parameter declares. Every launch was
checked: each kernel's parameter types against the types of the host
variables passed. The only mismatches were the two fixed in 2.0.0 (Granger
passed a `size_t` and MI its pitches as `size_t` for `int` parameters,
which read correctly only because the low half comes first on
little-endian machines); all launches now match.

## Decisions taken

1. **`gpuSolve` computes on the GPU** (it called base R's `qr()`, `qr.Q()`
   and `qr.R()` on the CPU and used the GPU only for the final step). It now
   uses NVIDIA's cuSOLVER: LU with partial pivoting (`getrf`/`getrs`) for a
   square `x`, as `solve()` uses LAPACK's, and Householder QR least squares
   (`geqrf`, `ormqr`, then a triangular solve) for an overdetermined one.
   An exactly singular square `x`, or a rank-deficient tall one, is an
   error. Matches `solve()` and `qr.coef()` to 1e-10 in double precision.
   RTX 2080 Ti, n = 4000: the inverse took 42.9 s before and 0.34 s now
   (`solve()`: 0.67 s); x b = y 0.13 s (`solve()`: 0.30 s).
2. **`beta` for the flexible clustering methods**, exposed on `gpuHclust`
   and `gpuDistClust` (lambda = beta, so ai = aj = (1 - beta)/2 as Lance &
   Williams define it), must be below 1. The default stays 0.5 for
   compatibility; the help documents Lance & Williams' recommended -0.25.
   Both methods match `lanceWilliams()` for beta = -0.25, 0 and 0.5.
