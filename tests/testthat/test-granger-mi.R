set.seed(6)
len <- 120
cause <- rnorm(len)
effect <- c(0, 0, 0.8 * cause[1:(len - 2)]) + rnorm(len, sd = 0.5)
noise <- rnorm(len)
x <- cbind(effect, cause, noise)

# fStatistics[i, j] answers "does column i Granger-cause column j?"
for (pr in precisions) {
  test_that(paste("gpuGranger matches an lm() F-test in", pr), {
    skip_if_no_gpu()
    for (lag in 1:2) {
      g <- gpuGranger(x, lag = lag, precision = pr)
      for (i in 1:3) for (j in 1:3) if (i != j) {
        ref <- grangerRef(x[, j], x[, i], lag)
        if (pr == "double") {
          expect_equal(g$fStatistics[i, j], unname(ref["f"]), tolerance = 1e-8)
          expect_equal(g$pValues[i, j], unname(ref["p"]), tolerance = 1e-6)
        } else {
          # near-zero F loses relative precision to float32 cancellation in
          # the residual sums of squares; compare on an absolute floor
          expect_lt(abs(g$fStatistics[i, j] - ref["f"]),
                    1e-3 * max(1, ref["f"]))
          expect_lt(abs(g$pValues[i, j] - ref["p"]), 1e-3)
        }
      }
    }
  })

  test_that(paste("gpuGranger x/y form agrees with the single-matrix form in",
                  pr), {
    skip_if_no_gpu()
    g <- gpuGranger(x, lag = 2, precision = pr)
    # several x columns, one y column
    gxy <- gpuGranger(x[, 2:3], x[, 1, drop = FALSE], lag = 2, precision = pr)
    expect_equal(gxy$fStatistics[, 1], g$fStatistics[2:3, 1],
                 tolerance = tolFor(pr), ignore_attr = TRUE)
    # one x column, several y columns
    gxy <- gpuGranger(x[, 1, drop = FALSE], x[, 2:3], lag = 2, precision = pr)
    expect_equal(gxy$fStatistics[1, ], g$fStatistics[1, 2:3],
                 tolerance = tolFor(pr), ignore_attr = TRUE)
  })
}

test_that("gpuGranger's result says which way it reads", {
  skip_if_no_gpu()
  g <- gpuGranger(x, lag = 2)
  expect_identical(dimnames(g$fStatistics),
                   list(cause = colnames(x), effect = colnames(x)))
  expect_true(all(is.na(diag(g$fStatistics))) && all(is.na(diag(g$pValues))))
  # the strong effect is cause -> effect, not the reverse
  expect_gt(g$fStatistics["cause", "effect"], 100)
  expect_lt(g$fStatistics["effect", "cause"], 5)
  gxy <- gpuGranger(x[, 2:3], x[, 1, drop = FALSE], lag = 2)
  expect_identical(dimnames(gxy$fStatistics),
                   list(cause = c("cause", "noise"), effect = "effect"))
  un <- gpuGranger(unname(x), lag = 1)
  expect_identical(rownames(un$fStatistics), c("x1", "x2", "x3"))
})

test_that("small Granger p-values are not rounded to zero", {
  skip_if_no_gpu()
  p <- gpuGranger(x, lag = 2, precision = "double")$pValues[2, 1]
  expect_gt(p, 0)
  expect_lt(p, 1e-10)
})

test_that("gpuMi is symmetric, non-negative and sees dependence", {
  skip_if_no_gpu()
  m <- gpuMi(x, bins = 6, splineOrder = 3)
  expect_equal(m, t(m), tolerance = tol, ignore_attr = TRUE)
  expect_true(all(m >= -tol))
  expect_gt(m["effect", "cause"], m["noise", "cause"])
})

test_that("gpuMi agrees in single and double precision", {
  skip_if_no_gpu()
  expect_equal(gpuMi(x, bins = 6, splineOrder = 3, precision = "single"),
               gpuMi(x, bins = 6, splineOrder = 3, precision = "double"),
               tolerance = tol)
})

for (pr in precisions) {
  test_that(paste("gpuMi reproduces Daub et al.'s worked example in", pr), {
    skip_if_no_gpu()
    # BMC Bioinformatics 5:118 (2004), section "Example": M = 3, k = 2
    ex <- cbind(x = c(0, .2, .4, .6, .8, 1))
    ey <- cbind(y = c(.8, 1, .6, .4, 0, .2))
    mi <- gpuMi(ex, ey, bins = 3, splineOrder = 2, precision = pr)[1, 1]
    expect_equal(mi, daubMI(ex[, 1], ey[, 1], 3, 2), tolerance = tolFor(pr))
    expect_equal(round(mi, 2), 0.45)                      # as printed
  })

  test_that(paste("gpuMi follows Daub et al.'s definition in", pr), {
    skip_if_no_gpu()
    set.seed(5)
    X <- matrix(rnorm(200 * 3), 200, 3)
    for (b in c(5, 10)) for (k in 1:4) {
      ref <- outer(1:3, 1:3, Vectorize(function(i, j)
        daubMI(X[, i], X[, j], b, k)))
      expect_equal(gpuMi(X, bins = b, splineOrder = k, precision = pr), ref,
                   tolerance = tolFor(pr), ignore_attr = TRUE,
                   label = sprintf("bins %d, order %d", b, k))
    }
  })
}

test_that("the reference reproduces the paper's own numbers", {
  ex <- c(0, .2, .4, .6, .8, 1)
  w <- bsplineWeights(ex, 3, 2)
  expect_equal(w, cbind(c(1, .6, .2, 0, 0, 0), c(0, .4, .8, .8, .4, 0),
                        c(0, 0, 0, .2, .6, 1)), ignore_attr = TRUE)
  expect_equal(colMeans(w) * 6, c(1.8, 2.4, 1.8))
  expect_equal(entropyBits(colMeans(w)),
               log2(10) - 0.6 * log2(3) - 0.4 * log2(4))
})
