set.seed(5)
n <- 200
d <- data.frame(a = rnorm(n), b = rnorm(n), g = gl(4, n / 4))
d$y <- 1 + 2 * d$a - 0.5 * d$b + as.integer(d$g) / 4 + rnorm(n, sd = 0.3)
d$z <- rbinom(n, 1, plogis(0.3 + d$a - d$b))
d$w <- runif(n, 0.5, 2)

test_that("gpuLm matches lm()", {
  skip_if_no_gpu()
  expect_equal(coef(gpuLm(y ~ a + b + g, d)), coef(lm(y ~ a + b + g, d)),
               tolerance = tol)
})

test_that("weighted gpuLm matches lm()", {
  skip_if_no_gpu()
  expect_equal(coef(gpuLm(y ~ a + b, d, weights = w)),
               coef(lm(y ~ a + b, d, weights = w)), tolerance = tol)
})

test_that("gpuLsfit matches lsfit()", {
  skip_if_no_gpu()
  X <- as.matrix(d[, c("a", "b")])
  expect_equal(gpuLsfit(X, d$y)$coefficients, lsfit(X, d$y)$coefficients,
               tolerance = tol, ignore_attr = TRUE)
})

test_that("gpuGlm matches glm() (gaussian and binomial)", {
  skip_if_no_gpu()
  expect_equal(coef(gpuGlm(y ~ a + b, data = d)),
               coef(glm(y ~ a + b, data = d)), tolerance = tol)
  expect_equal(coef(gpuGlm(z ~ a + b, binomial, d)),
               coef(glm(z ~ a + b, binomial, d)), tolerance = 1e-3)
})

test_that("gpuLsfit matches lsfit() past one 128-column block", {
  skip_if_no_gpu()
  # the blocked QR only updates to the right of a block when there is more
  # than one block of columns
  set.seed(8)
  X <- matrix(rnorm(400 * 150), 400, 150)
  yy <- drop(X %*% rnorm(150)) + rnorm(400)
  expect_equal(gpuLsfit(X, yy)$coefficients, lsfit(X, yy)$coefficients,
               tolerance = 1e-3, ignore_attr = TRUE)
})
