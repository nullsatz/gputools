# gpuCorPairs against the full matrix from cor(): the same pairs and values,
# whatever the tile size. Pairs within float error of the threshold may
# fall either way in single precision, so they are left out of the
# comparison.

# the reference edge list from cor(): pairs with strength >= threshold
refThreshold <- function(x, y, threshold, absolute, method) {
  r <- if (is.null(y)) cor(x, method = method) else cor(x, y, method = method)
  s <- if (absolute) abs(r) else r
  keep <- !is.na(s) & s >= threshold
  if (is.null(y)) keep[lower.tri(keep, diag = TRUE)] <- FALSE
  w <- which(keep, arr.ind = TRUE)
  w <- w[order(w[, 2], w[, 1]), , drop = FALSE]
  list(pairs = paste(w[, 1], w[, 2]), r = r[w], all = r)
}

# pairs whose strength is within eps of the threshold
borderline <- function(r, threshold, absolute, eps = 1e-4) {
  s <- if (absolute) abs(r) else r
  w <- which(!is.na(s) & abs(s - threshold) < eps, arr.ind = TRUE)
  paste(w[, 1], w[, 2])
}

for (pr in precisions) {
  test_that(paste("threshold pairs match cor() at any tile size in", pr), {
    skip_if_no_gpu()
    set.seed(41)
    x <- matrix(rnorm(40 * 30), 40, 30)
    x[, 5] <- x[, 4] + rnorm(40, sd = 0.2)       # some strong pairs
    x[, 20] <- -x[, 3] + rnorm(40, sd = 0.2)
    y <- matrix(rnorm(40 * 17), 40, 17)
    y[, 2] <- x[, 9] + rnorm(40, sd = 0.3)
    for (args in list(list(y = NULL, absolute = TRUE),
                      list(y = NULL, absolute = FALSE),
                      list(y = y, absolute = TRUE))) {
      for (method in c("pearson", "spearman")) {
        thr <- 0.3
        ref <- refThreshold(x, args$y, thr, args$absolute, method)
        skipPairs <- borderline(ref$all, thr, args$absolute)
        for (tile in c(1L, 7L, 16L, 2048L)) {
          got <- gpuCorPairs(x, args$y, threshold = thr, method = method,
                             absolute = args$absolute, tile = tile,
                             precision = pr)
          gotPairs <- paste(got$i, got$j)
          what <- paste(method, "tile", tile,
                        if (is.null(args$y)) "self" else "xy",
                        if (args$absolute) "abs" else "signed")
          expect_setequal(setdiff(gotPairs, skipPairs),
                          setdiff(ref$pairs, skipPairs))
          expect_equal(got$r, ref$all[cbind(got$i, got$j)],
                       tolerance = tolFor(pr), label = what)
          expect_false(is.unsorted(got$j + got$i / 1e6), label = what)
          if (is.null(args$y)) expect_true(all(got$i < got$j), label = what)
        }
      }
    }
  })

  test_that(paste("top pairs match cor() at any tile size in", pr), {
    skip_if_no_gpu()
    set.seed(42)
    x <- matrix(rnorm(25 * 23), 25, 23)
    y <- matrix(rnorm(25 * 9), 25, 9)
    for (self in c(TRUE, FALSE)) for (absolute in c(TRUE, FALSE)) {
      r <- if (self) cor(x) else cor(x, y)
      s <- if (absolute) abs(r) else r
      if (self) diag(s) <- NA
      for (k in c(1L, 3L, 40L)) for (tile in c(1L, 5L, 8L, 2048L)) {
        got <- gpuCorPairs(x, if (self) NULL else y, top = k,
                           absolute = absolute, tile = tile, precision = pr)
        what <- paste("k", k, "tile", tile, if (self) "self" else "xy",
                      if (absolute) "abs" else "signed")
        kk <- min(k, nrow(s) - self)
        expect_equal(as.vector(table(factor(got$j, seq_len(ncol(s))))),
                     rep(kk, ncol(s)), label = what)
        for (j in seq_len(ncol(s))) {
          want <- sort(s[, j], decreasing = TRUE)[seq_len(kk)]
          g <- got[got$j == j, ]
          gs <- if (absolute) abs(g$r) else g$r
          # the same strengths, strongest first (ties in float may swap
          # which column is listed, not how strong it is)
          expect_equal(gs, unname(want), tolerance = tolFor(pr),
                       label = paste(what, "column", j))
          expect_equal(g$r, r[cbind(g$i, j)], tolerance = tolFor(pr))
          if (self) expect_false(any(g$i == j))
        }
      }
    }
  })
}

test_that("gpuCorPairs names, tests and refuses as documented", {
  skip_if_no_gpu()
  set.seed(43)
  x <- matrix(rnorm(30 * 6), 30, 6, dimnames = list(NULL, letters[1:6]))
  x[, 2] <- x[, 1] + rnorm(30, sd = 0.1)
  got <- gpuCorPairs(x, threshold = 0.5, test = TRUE, precision = "double")
  expect_equal(got$x, letters[got$i])
  expect_equal(got$y, letters[got$j])
  ct <- cor.test(x[, got$i[1]], x[, got$j[1]])
  expect_equal(got$t[1], unname(ct$statistic), tolerance = 1e-8)
  expect_equal(got$p.value[1], ct$p.value, tolerance = 1e-8)

  # a constant column correlates with nothing
  x[, 3] <- 1
  all <- gpuCorPairs(x, threshold = 0, precision = "double")
  expect_false(any(all$i == 3 | all$j == 3))
  expect_equal(nrow(all), choose(5, 2))
  expect_false(3 %in% gpuCorPairs(x, top = 5)$i)

  expect_error(gpuCorPairs(x), "exactly one")
  expect_error(gpuCorPairs(x, threshold = 0.5, top = 2), "exactly one")
  expect_error(gpuCorPairs(x, top = 0), "positive")
  expect_error(gpuCorPairs(x, threshold = NA), "number")
  expect_error(gpuCorPairs(x, threshold = 0.5, method = "kendall"),
               "pearson")
  expect_error(gpuCorPairs(x, threshold = 0, limit = 3), "more than 3")
  expect_error(gpuCorPairs(x[1:2, ], threshold = 0), "3 rows")
  x[1, 1] <- NA
  expect_error(gpuCorPairs(x, threshold = 0.5), "missing")
  expect_error(gpuCorPairs(x[, 1:2], matrix(1, 5, 2), threshold = 0.5),
               "rows")
})
