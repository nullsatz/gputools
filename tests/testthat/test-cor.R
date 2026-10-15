set.seed(2)
x <- matrix(rnorm(50 * 6), 50, 6)
y <- matrix(rnorm(50 * 4), 50, 4)

for (pr in precisions) {
  test_that(paste("pearson matches cor() in", pr), {
    skip_if_no_gpu()
    expect_equal(gpuCor(x, precision = pr)$coefficients, cor(x),
                 tolerance = tolFor(pr), ignore_attr = TRUE)
    expect_equal(gpuCor(x, y, precision = pr)$coefficients, cor(x, y),
                 tolerance = tolFor(pr), ignore_attr = TRUE)
  })

  test_that(paste("pairwise.complete.obs matches cor() in", pr), {
    skip_if_no_gpu()
    xna <- x
    xna[c(3, 17, 40), 2] <- NA
    xna[c(5, 17), 5] <- NA
    expect_equal(gpuCor(xna, use = "pairwise.complete.obs",
                        precision = pr)$coefficients,
                 cor(xna, use = "pairwise.complete.obs"),
                 tolerance = tolFor(pr), ignore_attr = TRUE)
  })

  test_that(paste("kendall matches cor() in", pr), {
    skip_if_no_gpu()
    expect_equal(gpuCor(x, y, method = "kendall", precision = pr)$coefficients,
                 cor(x, y, method = "kendall"), tolerance = tolFor(pr),
                 ignore_attr = TRUE)
  })

  test_that(paste("kendall matches cor() (tau-b) with ties in", pr), {
    skip_if_no_gpu()
    set.seed(9)
    xt <- matrix(sample(1:4, 60 * 3, replace = TRUE), 60, 3)
    yt <- matrix(sample(1:5, 60 * 2, replace = TRUE), 60, 2)
    expect_equal(gpuCor(xt, yt, method = "kendall",
                        precision = pr)$coefficients,
                 cor(xt, yt, method = "kendall"), tolerance = tolFor(pr),
                 ignore_attr = TRUE)
  })

  test_that(paste("gpuTtest gives the correlation t-score in", pr), {
    skip_if_no_gpu()
    r <- c(0.1, -0.4, 0.75)
    n <- c(20, 30, 50)
    expect_equal(gpuTtest(n, r, precision = pr),
                 r * sqrt((n - 2) / (1 - r^2)), tolerance = tolFor(pr))
  })
}

test_that("kendall counts stay exact past 2^24 pairs", {
  skip_if_no_gpu()
  # 6000 samples = ~18M pairs, where float counting stopped being exact;
  # integer data is exact in either precision, so both must match
  set.seed(10)
  big <- matrix(sample(1:50, 6000 * 2, replace = TRUE), 6000, 2)
  ref <- cor(big[, 1], big[, 2], method = "kendall")
  for (pr in precisions)
    expect_equal(gpuCor(big[, 1, drop = FALSE], big[, 2, drop = FALSE],
                        method = "kendall", precision = pr)$coefficients[1, 1],
                 ref, tolerance = 1e-12)
})
