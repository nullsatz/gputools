# gpuCorPerm against the same permutations done by cor(): the same counts
# of permuted correlations at least as strong, and the same maxT adjusted
# p-values. In double precision they agree exactly; in single a
# permuted correlation within rounding of the observed one may count
# either way, so the counts may differ by a few.

permRef <- function(x, y, B, seed, method) {
  set.seed(seed)
  self <- is.null(y)
  if (self) y <- x
  n <- nrow(x)
  perms <- vapply(seq_len(B), function(b) sample.int(n), integer(n))
  r0 <- cor(x, y, method = method)
  if (self) diag(r0) <- NA
  counts <- matrix(0, ncol(x), ncol(y))
  maxima <- numeric(B)
  for (b in seq_len(B)) {
    rb <- cor(x, y[perms[, b], , drop = FALSE], method = method)
    if (self) diag(rb) <- NA
    counts <- counts + (abs(rb) >= abs(r0) - 1e-10)
    maxima[b] <- max(abs(rb), na.rm = TRUE)
  }
  adjusted <- (vapply(abs(r0), function(a) sum(maxima >= a - 1e-10), 0) + 1) /
    (B + 1)
  list(r = r0, p = (counts + 1) / (B + 1),
       adjusted = matrix(adjusted, nrow(r0)))
}

for (pr in precisions) {
  test_that(paste("gpuCorPerm matches the same permutations by cor() in", pr), {
    skip_if_no_gpu()
    set.seed(71)
    x <- matrix(rexp(12 * 9), 12, 9)
    y <- matrix(rexp(12 * 7), 12, 7)
    y[, 2] <- y[, 2] + 2 * x[, 4]
    B <- 300
    for (yy in list(NULL, y)) for (method in c("pearson", "spearman")) {
      ref <- permRef(x, yy, B, 5, method)
      set.seed(5)
      got <- gpuCorPerm(x, yy, B = B, method = method, precision = pr)
      what <- paste(method, if (is.null(yy)) "self" else "xy")
      expect_equal(got$coefficients, ref$r, tolerance = tolFor(pr),
                   label = what)
      slack <- if (pr == "double") 0 else 3 / (B + 1)
      expect_lte(max(abs(got$p.value - ref$p), na.rm = TRUE), slack + 1e-12,
                 label = what)
      expect_lte(max(abs(got$p.adjusted - ref$adjusted), na.rm = TRUE),
                 slack + 1e-12, label = what)
      expect_true(all(got$p.adjusted >= got$p.value, na.rm = TRUE))
      if (is.null(yy)) expect_true(all(is.na(diag(got$p.value))))
    }
  })
}

test_that("gpuCorPerm finds a real association and refuses as documented", {
  skip_if_no_gpu()
  set.seed(72)
  x <- matrix(rnorm(30 * 20), 30, 20, dimnames = list(NULL, letters[1:20]))
  y <- matrix(rnorm(30 * 5), 30, 5)
  y[, 3] <- x[, 2] + rnorm(30, sd = 0.3)
  res <- gpuCorPerm(x, y, B = 999)
  expect_equal(unname(res$p.value[2, 3]), 1 / 1000)
  expect_lt(res$p.adjusted[2, 3], 0.01)
  expect_gt(min(res$p.adjusted[-2, ]), 0.01)
  expect_equal(rownames(res$p.value), letters[1:20])
  expect_equal(res$permutations, 999L)

  # a constant column has no tests
  x[, 5] <- 1
  res <- gpuCorPerm(x, y, B = 50)
  expect_true(all(is.na(res$p.value[5, ])))
  expect_true(all(is.na(res$p.adjusted[5, ])))

  expect_error(gpuCorPerm(x, B = 0), "positive")
  expect_error(gpuCorPerm(x, method = "kendall"), "pearson")
  expect_error(gpuCorPerm(x[1:2, ]), "3 rows")
  expect_error(gpuCorPerm(x, y[1:5, ]), "rows")
  x[1, 1] <- NA
  expect_error(gpuCorPerm(x), "missing")
})
