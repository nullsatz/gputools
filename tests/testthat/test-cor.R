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

  test_that(paste("gpuTtest gives the correlation t-score in", pr), {
    skip_if_no_gpu()
    r <- c(0.1, -0.4, 0.75)
    n <- c(20, 30, 50)
    expect_equal(gpuTtest(n, r, precision = pr),
                 r * sqrt((n - 2) / (1 - r^2)), tolerance = tolFor(pr))
  })
}
