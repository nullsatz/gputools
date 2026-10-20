set.seed(6)
len <- 120
cause <- rnorm(len)
effect <- c(0, 0, 0.8 * cause[1:(len - 2)]) + rnorm(len, sd = 0.5)
noise <- rnorm(len)
x <- cbind(effect, cause, noise)

# fStatistics[i, j] answers "does column i Granger-cause column j?"
# The tests compute in double whatever the precision argument says
for (pr in precisions) {
  test_that(paste("gpuGranger matches an lm() F-test with precision", pr), {
    skip_if_no_gpu()
    for (lag in 1:2) {
      g <- gpuGranger(x, lag = lag, precision = pr)
      for (i in 1:3) for (j in 1:3) if (i != j) {
        ref <- grangerRef(x[, j], x[, i], lag)
        expect_equal(g$fStatistics[i, j], unname(ref["f"]), tolerance = 1e-8)
        expect_equal(g$pValues[i, j], unname(ref["p"]), tolerance = 1e-8)
      }
    }
  })

  test_that(paste("gpuGranger x/y form agrees with the single-matrix form,",
                  pr), {
    skip_if_no_gpu()
    g <- gpuGranger(x, lag = 2, precision = pr)
    # several x columns, one y column
    gxy <- gpuGranger(x[, 2:3], x[, 1, drop = FALSE], lag = 2, precision = pr)
    expect_equal(gxy$fStatistics[, 1], g$fStatistics[2:3, 1],
                 tolerance = 1e-10, ignore_attr = TRUE)
    # one x column, several y columns
    gxy <- gpuGranger(x[, 1, drop = FALSE], x[, 2:3], lag = 2, precision = pr)
    expect_equal(gxy$fStatistics[1, ], g$fStatistics[1, 2:3],
                 tolerance = 1e-10, ignore_attr = TRUE)
  })
}

test_that("gpuGranger is accurate on long random walks and trends", {
  skip_if_no_gpu()
  set.seed(7)
  n <- 5000
  walks <- apply(matrix(rnorm(n * 4), n, 4), 2, cumsum)
  walks[2:n, 3] <- walks[2:n, 3] + 0.05 * walks[1:(n - 1), 1]
  drift <- apply(1 + matrix(rnorm(n * 3), n, 3), 2, cumsum)
  trend <- 0.01 * seq_len(n) + matrix(rnorm(n * 3), n, 3)
  for (lag in c(1, 3)) for (x in list(walks, drift, trend)) {
    g <- expect_no_warning(gpuGranger(x, lag = lag))
    p <- ncol(x)
    for (i in 1:p) for (j in 1:p) if (i != j) {
      ref <- grangerRef(x[, j], x[, i], lag)
      expect_equal(g$fStatistics[i, j], unname(ref["f"]), tolerance = 1e-7,
                   info = paste("pair", i, j, "lag", lag))
    }
  }
})

test_that("gpuGranger resolves a cause nearly equal to the effect", {
  skip_if_no_gpu()
  set.seed(50)
  e <- cumsum(rnorm(2000))
  x <- cbind(e, e + 1e-6 * rnorm(2000), cumsum(rnorm(2000)))
  g <- expect_no_warning(gpuGranger(x, lag = 2))
  for (ij in list(c(2, 1), c(3, 1))) {
    ref <- grangerRef(x[, ij[2]], x[, ij[1]], 2, tol = 1e-12)
    expect_equal(g$fStatistics[ij[1], ij[2]], unname(ref["f"]),
                 tolerance = 1e-6)
  }
})

test_that("gpuGranger flags what it cannot resolve, and only that", {
  skip_if_no_gpu()
  set.seed(51)
  x <- apply(matrix(rnorm(150 * 3), 150, 3), 2, cumsum)
  # a cause that is the effect: nothing of its lags is left beside them
  expect_warning(g <- gpuGranger(x, x[, 2, drop = FALSE], lag = 2),
                 "1 pair\\(s\\) are too ill-conditioned")
  expect_true(is.na(g$fStatistics[2, 1]))
  expect_true(is.finite(g$fStatistics[1, 1]) && is.finite(g$fStatistics[3, 1]))
  # a constant series has no tests, and no warning
  xc <- x
  xc[, 2] <- 4
  g <- expect_no_warning(gpuGranger(xc, lag = 2))
  expect_true(all(is.na(g$fStatistics[2, ])) && all(is.na(g$fStatistics[, 2])))
  expect_true(is.finite(g$fStatistics[1, 3]))
  # series integrated three times at a long lag: every pair is either
  # flagged (NA, with the warning) or within 0.1 percent of least squares
  set.seed(1)
  x3 <- matrix(rnorm(3000 * 3), 3000, 3)
  for (k in 1:3) x3 <- apply(x3, 2, cumsum)
  warned <- 0
  g <- withCallingHandlers(gpuGranger(x3, lag = 5),
                           warning = function(w) {
                             warned <<- warned + 1
                             invokeRestart("muffleWarning")
                           })
  off <- row(g$fStatistics) != col(g$fStatistics)
  flagged <- sum(is.na(g$fStatistics[off]))
  expect_equal(warned, as.integer(flagged > 0))
  for (i in 1:3) for (j in 1:3) if (i != j && is.finite(g$fStatistics[i, j])) {
    ref <- grangerRef(x3[, j], x3[, i], 5, tol = 1e-12)
    expect_equal(g$fStatistics[i, j], unname(ref["f"]), tolerance = 1e-3,
                 info = paste("pair", i, j))
  }
})

test_that("gpuGranger agrees with lm() across several tiles of effects", {
  skip_if_no_gpu()
  # n = 20,000 at lag 3 puts about 34 effects in a tile, so 50 series
  # take two tiles: the second tile's indexing, its diagonal and the x/y
  # form on a subset are all exercised
  set.seed(8)
  n <- 20000
  x <- matrix(rnorm(n * 50), n, 50)
  x[2:n, 45] <- x[2:n, 45] + 0.1 * x[1:(n - 1), 3]
  g <- expect_no_warning(gpuGranger(x, lag = 3))
  expect_true(all(is.na(diag(g$fStatistics))))
  expect_true(all(is.finite(g$fStatistics[row(g$fStatistics) != col(g$fStatistics)])))
  pairs <- rbind(c(3, 45), c(45, 3), c(1, 50), c(50, 1), c(34, 35), c(35, 34),
                 c(10, 40), c(40, 49), c(49, 48), c(2, 36))
  for (r in seq_len(nrow(pairs))) {
    i <- pairs[r, 1]; j <- pairs[r, 2]
    ref <- grangerRef(x[, j], x[, i], 3)
    expect_equal(g$fStatistics[i, j], unname(ref["f"]), tolerance = 1e-7,
                 info = paste("pair", i, j))
  }
  expect_gt(g$fStatistics[3, 45], 20)
  gxy <- gpuGranger(x[, 1:5], x[, 40:50], lag = 3)
  expect_equal(gxy$fStatistics, g$fStatistics[1:5, 40:50], tolerance = 1e-10,
               ignore_attr = TRUE)
})

test_that("gpuGranger checks its arguments", {
  skip_if_no_gpu()
  expect_error(gpuGranger(x, lag = 0), "positive whole number")
  expect_error(gpuGranger(x, lag = 1.5), "positive whole number")
  expect_error(gpuGranger(x, lag = 17), "at most 16")
  expect_error(gpuGranger(x, lag = 1e10), "at most 16")
  expect_error(gpuGranger(x[, 0], lag = 1), "at least one column")
  expect_error(gpuGranger(x[1:7, ], lag = 2), "too short")
  expect_error(gpuGranger(x, x[1:100, ], lag = 2), "different numbers of rows")
  expect_equal(dim(gpuGranger(x, lag = 16)$fStatistics), c(3, 3))
})

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

test_that("gpuMi gives a constant variable exactly 0, and checks its arguments", {
  skip_if_no_gpu()
  set.seed(9)
  X <- matrix(rnorm(100 * 3), 100, 3)
  X[, 2] <- 3
  m <- gpuMi(X)
  expect_false(anyNA(m))
  expect_true(all(m[2, ] == 0) && all(m[, 2] == 0))
  expect_gt(m[1, 1], 0)
  expect_true(all(m >= 0))
  expect_equal(gpuMi(X, X), m, ignore_attr = TRUE)
  expect_error(gpuMi(X, bins = 3, splineOrder = 4), "splineOrder <= bins")
  expect_error(gpuMi(X, bins = 2.5), "whole numbers")
  expect_error(gpuMi(X, bins = 65), "at most 64")
  expect_error(gpuMi(X, X[1:50, ]), "different numbers of rows")
  expect_error(gpuMi(X[1, , drop = FALSE]), "at least 2 rows")
})

test_that("gpuMi in single precision keeps knot-valued data in their bins", {
  skip_if_no_gpu()
  # discrete data sit exactly on the knots; weights rounded to single
  # once moved values into the neighbouring bin, so single and double
  # disagreed by far more than single's rounding
  set.seed(10)
  X <- matrix(sample(0:9, 300 * 4, replace = TRUE), 300, 4)
  for (b in c(5, 10, 20)) {
    expect_equal(gpuMi(X, bins = b, splineOrder = 3, precision = "single"),
                 gpuMi(X, bins = b, splineOrder = 3, precision = "double"),
                 tolerance = 1e-5, label = paste("bins", b))
  }
})
