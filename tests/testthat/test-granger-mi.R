set.seed(6)
len <- 120
cause <- rnorm(len)
effect <- c(0, 0, 0.8 * cause[1:(len - 2)]) + rnorm(len, sd = 0.5)
noise <- rnorm(len)
x <- cbind(effect, cause, noise)

# fStatistics[i, j] answers "does column i Granger-cause column j?"
test_that("gpuGranger matches an lm() F-test", {
  skip_if_no_gpu()
  for (lag in 1:2) {
    g <- gpuGranger(x, lag = lag)
    for (i in 1:3) for (j in 1:3) if (i != j) {
      ref <- grangerRef(x[, j], x[, i], lag)
      # near-zero F loses relative precision to float32 cancellation in the
      # residual sums of squares; compare on an absolute floor
      expect_lt(abs(g$fStatistics[i, j] - ref["f"]), 1e-3 * max(1, ref["f"]))
      expect_lt(abs(g$pValues[i, j] - ref["p"]), 1e-3)
    }
  }
})

test_that("gpuGranger x/y form agrees with the single-matrix form", {
  skip_if_no_gpu()
  g <- gpuGranger(x, lag = 2)
  # several x columns, one y column
  gxy <- gpuGranger(x[, 2:3], x[, 1, drop = FALSE], lag = 2)
  expect_equal(gxy$fStatistics[, 1], g$fStatistics[2:3, 1], tolerance = tol,
               ignore_attr = TRUE)
  # one x column, several y columns
  gxy <- gpuGranger(x[, 1, drop = FALSE], x[, 2:3], lag = 2)
  expect_equal(gxy$fStatistics[1, ], g$fStatistics[1, 2:3], tolerance = tol,
               ignore_attr = TRUE)
})

test_that("gpuMi is symmetric, non-negative and sees dependence", {
  skip_if_no_gpu()
  m <- gpuMi(x, bins = 6, splineOrder = 3)
  expect_equal(m, t(m), tolerance = tol, ignore_attr = TRUE)
  expect_true(all(m >= -tol))
  expect_gt(m["effect", "cause"], m["noise", "cause"])
})
