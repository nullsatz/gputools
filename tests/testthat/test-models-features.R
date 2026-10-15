# lm()/glm()/lsfit() features the GPU versions carry over, each checked
# against base R in double precision
set.seed(16)
n <- 80
fd <- data.frame(a = rnorm(n), b = rnorm(n), g = gl(4, n / 4),
                 off = runif(n), w = runif(n, 0.5, 2))
fd$y <- 1 + fd$a - fd$b + rnorm(n, sd = 0.2)
fd$y2 <- 2 - fd$a + rnorm(n, sd = 0.2)
fd$count <- rpois(n, exp(0.5 + 0.3 * fd$a))
tol8 <- 1e-8

expectSameFit <- function(g, ref) {
  expect_equal(coef(g), coef(ref), tolerance = tol8)
  expect_equal(unname(residuals(g)), unname(residuals(ref)), tolerance = tol8)
  expect_equal(unname(fitted(g)), unname(fitted(ref)), tolerance = tol8)
}

test_that("gpuLm: offsets, zero weights, subsets, missing values", {
  skip_if_no_gpu()
  expectSameFit(gpuLm(y ~ a + offset(off), fd, useSingle = FALSE),
                lm(y ~ a + offset(off), fd))
  wz <- replace(fd$w, c(3, 17), 0)
  expectSameFit(gpuLm(y ~ a + b, fd, weights = wz, useSingle = FALSE),
                lm(y ~ a + b, fd, weights = wz))
  expectSameFit(gpuLm(y ~ a + g, fd, subset = a > -1, useSingle = FALSE),
                lm(y ~ a + g, fd, subset = a > -1))
  dna <- fd
  dna$a[c(4, 9)] <- NA
  expectSameFit(gpuLm(y ~ a + b, dna, useSingle = FALSE), lm(y ~ a + b, dna))
})

test_that("gpuLm: summaries, model frames and what it returns", {
  skip_if_no_gpu()
  g <- gpuLm(y ~ a + g, fd, x = TRUE, y = TRUE, useSingle = FALSE)
  ref <- lm(y ~ a + g, fd, x = TRUE, y = TRUE)
  expect_equal(coef(summary(g)), coef(summary(ref)), tolerance = 1e-6)
  expect_equal(g$x, ref$x)
  expect_equal(g$y, ref$y)
  expect_equal(predict(g, fd[1:5, ]), predict(ref, fd[1:5, ]), tolerance = tol8)
  expect_equal(gpuLm(y ~ a, fd, method = "model.frame"),
               lm(y ~ a, fd, method = "model.frame"))
  expect_warning(gpuLm(y ~ a, fd, method = "chol", useSingle = FALSE),
                 "not supported")
})

test_that("gpuLm: multiple responses, empty and intercept-only models", {
  skip_if_no_gpu()
  g <- gpuLm(cbind(y, y2) ~ a + b, fd, useSingle = FALSE)
  ref <- lm(cbind(y, y2) ~ a + b, fd)
  expect_equal(coef(g), coef(ref), tolerance = tol8)
  expect_equal(unname(residuals(g)), unname(residuals(ref)), tolerance = tol8)
  expect_equal(coef(gpuLm(y ~ 0, fd, useSingle = FALSE)), coef(lm(y ~ 0, fd)))
  expect_equal(coef(gpuLm(y ~ 1, fd, useSingle = FALSE)), coef(lm(y ~ 1, fd)),
               tolerance = tol8)
  e <- gpuLm(y ~ 0 + offset(off), fd, useSingle = FALSE)
  expect_equal(unname(fitted(e)), fd$off)
})

test_that("gpuLm.fit's own checks", {
  skip_if_no_gpu()
  X <- cbind(1, fd$a)
  expect_error(gpuLm.fit(fd$a, fd$y), "matrix")
  expect_error(gpuLm.fit(X[0, , drop = FALSE], numeric(0)), "0 \\(non-NA\\)")
  expect_error(gpuLm.fit(X, fd$y[-1]), "incompatible")
  expect_error(gpuLm.fit(X, fd$y, w = -fd$w), "negative")
  expect_length(gpuLm.fit(X[, 0, drop = FALSE], fd$y)$coefficients, 0)
  expect_warning(gpuLm.fit(X, fd$y, useSingle = FALSE, junk = 1), "disregarded")
})

test_that("gpuLsfit: weights, no intercept, missing values, several y", {
  skip_if_no_gpu()
  X <- as.matrix(fd[, c("a", "b")])
  for (args in list(list(wt = fd$w), list(intercept = FALSE))) {
    g <- do.call(gpuLsfit, c(list(X, fd$y, useSingle = FALSE), args))
    ref <- do.call(lsfit, c(list(X, fd$y), args))
    expect_equal(g$coefficients, ref$coefficients, tolerance = tol8)
    expect_equal(g$residuals, ref$residuals, tolerance = tol8)
  }
  Xna <- X
  Xna[5, 1] <- NA
  expect_warning(g <- gpuLsfit(Xna, fd$y, useSingle = FALSE), "missing")
  ref <- suppressWarnings(lsfit(Xna, fd$y))
  expect_equal(g$coefficients, ref$coefficients, tolerance = tol8)
  Y <- cbind(fd$y, fd$y2)
  expect_equal(gpuLsfit(X, Y, useSingle = FALSE)$coefficients,
               lsfit(X, Y)$coefficients, tolerance = tol8, ignore_attr = TRUE)
})

test_that("gpuGlm: Poisson with offsets, weights and starting values", {
  skip_if_no_gpu()
  fit <- function(f, ...) f(count ~ a + b + offset(log(off + 1)), poisson,
                           fd, ...)
  for (args in list(list(), list(weights = fd$w),
                    list(start = c(0.4, 0.3, 0)),
                    list(mustart = rep(mean(fd$count) + 0.5, n)),
                    list(etastart = rep(0.5, n)))) {
    g <- do.call(fit, c(list(gpuGlm), args, list(useSingle = FALSE)))
    ref <- do.call(fit, c(list(glm), args))
    expect_equal(coef(g), coef(ref), tolerance = 1e-6)
    expect_equal(deviance(g), deviance(ref), tolerance = 1e-6)
  }
})

test_that("gpuGlm: model frames, controls and non-convergence", {
  skip_if_no_gpu()
  expect_equal(gpuGlm(count ~ a, poisson, fd, method = "model.frame"),
               glm(count ~ a, poisson, fd, method = "model.frame"))
  expect_warning(gpuGlm(count ~ a + b, poisson, fd, useSingle = FALSE,
                        control = gpuGlm.control(FALSE, maxit = 1)),
                 "converge")
  g <- gpuGlm(count ~ a, poisson, fd, x = TRUE, y = FALSE, useSingle = FALSE)
  expect_equal(g$x, model.matrix(~ a, fd), ignore_attr = TRUE)
  expect_null(g$y)
  expect_equal(gpuGlm.control(TRUE)$epsilon, 1e-5)
})
