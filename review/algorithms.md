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
