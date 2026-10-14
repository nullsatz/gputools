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
