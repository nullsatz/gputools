set.seed(2)
x <- matrix(rnorm(50 * 6), 50, 6)
y <- matrix(rnorm(50 * 4), 50, 4)

test_that("pearson matches cor()", {
  skip_if_no_gpu()
  expect_equal(gpuCor(x)$coefficients, cor(x), tolerance = tol,
               ignore_attr = TRUE)
  expect_equal(gpuCor(x, y)$coefficients, cor(x, y), tolerance = tol,
               ignore_attr = TRUE)
})

test_that("pairwise.complete.obs matches cor()", {
  skip_if_no_gpu()
  xna <- x
  xna[c(3, 17, 40), 2] <- NA
  xna[c(5, 17), 5] <- NA
  expect_equal(gpuCor(xna, use = "pairwise.complete.obs")$coefficients,
               cor(xna, use = "pairwise.complete.obs"), tolerance = tol,
               ignore_attr = TRUE)
})

test_that("kendall matches cor()", {
  skip_if_no_gpu()
  expect_equal(gpuCor(x, y, method = "kendall")$coefficients,
               cor(x, y, method = "kendall"), tolerance = tol,
               ignore_attr = TRUE)
})

test_that("gpuTtest gives the correlation t-score", {
  skip_if_no_gpu()
  r <- c(0.1, -0.4, 0.75)
  n <- c(20, 30, 50)
  expect_equal(gpuTtest(n, r), r * sqrt((n - 2) / (1 - r^2)), tolerance = tol)
})
