set.seed(5)
n <- 200
d <- data.frame(a = rnorm(n), b = rnorm(n), g = gl(4, n / 4))
d$y <- 1 + 2 * d$a - 0.5 * d$b + as.integer(d$g) / 4 + rnorm(n, sd = 0.3)
d$z <- rbinom(n, 1, plogis(0.3 + d$a - d$b))
d$w <- runif(n, 0.5, 2)

for (pr in precisions) {
  single <- pr == "single"
  # double: as close as the different QR orderings allow; glm iterates
  # to a convergence tolerance, so it gets a looser one
  tolLm <- if (single) tol else 1e-8
  tolGlm <- if (single) 1e-3 else 1e-6

  test_that(paste("gpuLm matches lm() in", pr), {
    skip_if_no_gpu()
    expect_equal(coef(gpuLm(y ~ a + b + g, d, useSingle = single)),
                 coef(lm(y ~ a + b + g, d)), tolerance = tolLm)
  })

  test_that(paste("weighted gpuLm matches lm() in", pr), {
    skip_if_no_gpu()
    expect_equal(coef(gpuLm(y ~ a + b, d, weights = w, useSingle = single)),
                 coef(lm(y ~ a + b, d, weights = w)), tolerance = tolLm)
  })

  test_that(paste("gpuLsfit matches lsfit() in", pr), {
    skip_if_no_gpu()
    X <- as.matrix(d[, c("a", "b")])
    expect_equal(gpuLsfit(X, d$y, useSingle = single)$coefficients,
                 lsfit(X, d$y)$coefficients, tolerance = tolLm,
                 ignore_attr = TRUE)
  })

  test_that(paste("gpuGlm matches glm() (gaussian and binomial) in", pr), {
    skip_if_no_gpu()
    expect_equal(coef(gpuGlm(y ~ a + b, data = d, useSingle = single)),
                 coef(glm(y ~ a + b, data = d)), tolerance = tolLm)
    expect_equal(coef(gpuGlm(z ~ a + b, binomial, d, useSingle = single)),
                 coef(glm(z ~ a + b, binomial, d)), tolerance = tolGlm)
  })

  test_that(paste("gpuLsfit matches lsfit() past one 128-column block in",
                  pr), {
    skip_if_no_gpu()
    # the blocked QR only updates to the right of a block when there is
    # more than one block of columns
    set.seed(8)
    X <- matrix(rnorm(400 * 150), 400, 150)
    yy <- drop(X %*% rnorm(150)) + rnorm(400)
    expect_equal(gpuLsfit(X, yy, useSingle = single)$coefficients,
                 lsfit(X, yy)$coefficients,
                 tolerance = if (single) 1e-3 else 1e-8, ignore_attr = TRUE)
  })
}

test_that("the gputools.precision option sets the default", {
  skip_if_no_gpu()
  old <- options(gputools.precision = "double")
  on.exit(options(old))
  expect_equal(coef(gpuLm(y ~ a + b, d)), coef(lm(y ~ a + b, d)),
               tolerance = 1e-8)
  expect_equal(gpuQr(cbind(d$a, d$b))$rank, 2L)
})
