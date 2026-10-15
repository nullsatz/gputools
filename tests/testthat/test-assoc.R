# gpuAssoc against lm(): every pair's slope, t and p-value as
# summary(lm(y[, j] ~ x[, i] + covariates)) reports them.

lmScan <- function(x, y, covariates) {
  out <- expand.grid(i = seq_len(ncol(x)), j = seq_len(ncol(y)))
  stats <- t(mapply(function(i, j) {
    fit <- if (is.null(covariates)) lm(y[, j] ~ x[, i])
           else lm(y[, j] ~ x[, i] + covariates)
    summary(fit)$coefficients[2, c(1, 3, 4)]
  }, out$i, out$j))
  out$beta <- stats[, 1]
  out$t <- stats[, 2]
  out$p.value <- stats[, 3]
  out
}

assocData <- function(seed) {
  set.seed(seed)
  n <- 60
  x <- matrix(rbinom(n * 23, 2, 0.4), n, 23)
  storage.mode(x) <- "double"
  cv <- cbind(age = rnorm(n, 50, 10), sex = rbinom(n, 1, 0.5))
  y <- matrix(rnorm(n * 11), n, 11) + 0.02 * cv[, 1]
  y[, 3] <- y[, 3] + 2 * x[, 5]          # real associations
  y[, 9] <- y[, 9] - 1.5 * x[, 17] + 0.5 * cv[, 2]
  list(x = x, y = y, cv = cv)
}

for (pr in precisions) {
  test_that(paste("gpuAssoc matches lm() in", pr), {
    skip_if_no_gpu()
    d <- assocData(51)
    for (cv in list(NULL, d$cv, d$cv[, 1, drop = FALSE])) {
      ref <- lmScan(d$x, d$y, cv)
      for (tile in c(1L, 4L, 2048L)) {
        # every test, so all of them can be compared
        got <- gpuAssoc(d$x, d$y, covariates = cv, pvalue = 1, tile = tile,
                        precision = pr)
        expect_equal(nrow(got), nrow(ref))
        m <- merge(got, ref, by = c("i", "j"), suffixes = c("", ".lm"))
        expect_equal(m$beta, m$beta.lm, tolerance = tolFor(pr))
        expect_equal(m$t, m$t.lm, tolerance = tolFor(pr))
        expect_equal(m$p.value, m$p.value.lm, tolerance = tolFor(pr))
        expect_equal(got$fdr, p.adjust(got$p.value, "BH"),
                     tolerance = 1e-12)

        # a threshold reports exactly the tests at or below it
        cut <- 0.01
        sig <- gpuAssoc(d$x, d$y, covariates = cv, pvalue = cut, tile = tile,
                        precision = pr)
        want <- ref[ref$p.value <= cut, ]
        clear <- abs(log(want$p.value / cut)) > 1e-3   # not on the edge
        expect_true(all(paste(want$i, want$j)[clear] %in%
                        paste(sig$i, sig$j)))
        expect_true(all(sig$p.value <= cut))
        expect_false(is.unsorted(sig$p.value))
        # BH over the reported p-values only can overstate, not understate
        expect_true(all(sig$fdr >= (1 - tolFor(pr)) *
          p.adjust(ref$p.value, "BH")[
            match(paste(sig$i, sig$j), paste(ref$i, ref$j))]))

        # top: each y column's most significant x columns
        tp <- gpuAssoc(d$x, d$y, covariates = cv, top = 2, tile = tile,
                       precision = pr)
        expect_equal(nrow(tp), 2 * ncol(d$y))
        for (j in seq_len(ncol(d$y))) {
          want <- sort(ref$p.value[ref$j == j])[1:2]
          expect_equal(tp$p.value[tp$j == j], want, tolerance = tolFor(pr))
        }
      }
    }
  })
}

test_that("gpuAssoc finds the planted associations, names and refuses", {
  skip_if_no_gpu()
  d <- assocData(52)
  colnames(d$x) <- paste0("snp", 1:23)
  colnames(d$y) <- paste0("gene", 1:11)
  got <- gpuAssoc(d$x, d$y, d$cv, pvalue = 1e-6)
  expect_true(all(c("snp5 gene3", "snp17 gene9") %in%
                  paste(got$x, got$y)))
  expect_equal(got$x, colnames(d$x)[got$i])
  expect_equal(got$y, colnames(d$y)[got$j])

  # a column the covariates explain, or a constant one, has no test
  x <- d$x
  x[, 2] <- d$cv[, 1] * 2 + 1
  x[, 4] <- 1
  all <- gpuAssoc(x, d$y, d$cv, pvalue = 1, precision = "double")
  expect_false(any(all$i %in% c(2, 4)))
  expect_equal(nrow(all), 21 * 11)

  expect_error(gpuAssoc(d$x, d$y[1:5, ]), "rows")
  expect_error(gpuAssoc(d$x, d$y, d$cv[1:5, ]), "rows")
  expect_error(gpuAssoc(d$x, d$y, cbind(d$cv, d$cv[, 1])), "collinear")
  expect_error(gpuAssoc(d$x, d$y, pvalue = 0), "pvalue")
  expect_error(gpuAssoc(d$x, d$y, top = 0), "positive")
  expect_error(gpuAssoc(d$x, d$y, pvalue = 1, limit = 10), "more than 10")
  expect_error(gpuAssoc(d$x[1:3, ], d$y[1:3, ], d$cv[1:3, ]), "too few")
  d$y[1, 1] <- NA
  expect_error(gpuAssoc(d$x, d$y), "missing")
})
