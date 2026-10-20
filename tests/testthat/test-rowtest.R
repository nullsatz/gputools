# gpuRowTtest against t.test() row by row, and its permutation p-values
# against the same permutations computed in R.

rowRef <- function(x, group, var.equal, B, seed) {
  g <- as.factor(group)
  in1 <- g == levels(g)[1]
  tt <- function(lab, i) {
    unname(t.test(x[i, lab], x[i, !lab], var.equal = var.equal)$statistic)
  }
  t0 <- vapply(seq_len(nrow(x)), function(i) tt(in1, i), 0)
  out <- list(t = t0, df = vapply(seq_len(nrow(x)), function(i)
    unname(t.test(x[i, in1], x[i, !in1], var.equal = var.equal)$parameter), 0),
    p = vapply(seq_len(nrow(x)), function(i)
      t.test(x[i, in1], x[i, !in1], var.equal = var.equal)$p.value, 0))
  if (B > 0) {
    set.seed(seed)
    n <- ncol(x)
    tb <- matrix(0, nrow(x), B)
    for (b in seq_len(B)) {
      lab <- in1[sample.int(n)]
      tb[, b] <- vapply(seq_len(nrow(x)), function(i) tt(lab, i), 0)
    }
    out$pPerm <- (rowSums(abs(tb) >= abs(t0) * (1 - 1e-10)) + 1) / (B + 1)
    maxima <- apply(abs(tb), 2, max)
    out$pAdj <- (vapply(abs(t0), function(a) sum(maxima >= a * (1 - 1e-10)),
                        0) + 1) / (B + 1)
  }
  out
}

for (pr in precisions) {
  test_that(paste("gpuRowTtest matches t.test and the same permutations in", pr), {
    skip_if_no_gpu()
    set.seed(81)
    x <- matrix(rexp(15 * 11, 0.1) + 100, 15, 11)   # far from 0, skewed
    group <- c("b", "a", "b", "b", "a", "a", "b", "a", "b", "a", "b")
    x[2, group == "a"] <- x[2, group == "a"] + 40
    B <- 200
    for (ve in c(FALSE, TRUE)) {
      ref <- rowRef(x, group, ve, B, seed = 3)
      set.seed(3)
      got <- gpuRowTtest(x, group, var.equal = ve, B = B, precision = pr)
      what <- if (ve) "pooled" else "Welch"
      expect_equal(got$statistic, ref$t, tolerance = 1e-10, label = what)
      expect_equal(got$df, ref$df, tolerance = 1e-10, label = what)
      expect_equal(got$p.value, ref$p, tolerance = 1e-10, label = what)
      slack <- if (pr == "double") 0 else 2 / (B + 1)
      expect_lte(max(abs(got$p.perm - ref$pPerm)), slack + 1e-12, label = what)
      expect_lte(max(abs(got$p.adjusted - ref$pAdj)), slack + 1e-12,
                 label = what)
      expect_true(all(got$p.adjusted >= got$p.perm))
    }
  })
}

test_that("gpuRowTtest handles many permutations in blocks", {
  skip_if_no_gpu()
  # 40,000 rows make a block 838 permutations, so 2,000 take three
  set.seed(82)
  x <- matrix(rnorm(40000 * 20), 40000, 20)
  group <- rep(1:2, 10)
  x[1, group == 1] <- x[1, group == 1] + 10
  res <- gpuRowTtest(x, group, B = 2000, precision = "double")
  # only the observed labelling (or its mirror) reaches row 1's t, and of
  # choose(20, 10) labellings a few of 2,000 draws at most will be one
  expect_lte(res$p.perm[1], 3 / 2001)
  expect_lte(res$p.adjusted[1], 3 / 2001)
  expect_gt(median(res$p.adjusted[-1]), 0.5)
  expect_true(all(res$p.adjusted >= res$p.perm))
})

test_that("gpuRowTtest names, constant rows and refusals", {
  skip_if_no_gpu()
  set.seed(83)
  x <- matrix(rnorm(6 * 8), 6, 8, dimnames = list(letters[1:6], NULL))
  x[3, ] <- 5
  group <- factor(rep(c("x", "y"), 4))
  res <- gpuRowTtest(x, group, B = 20)
  expect_equal(rownames(res), letters[1:6])
  expect_true(is.nan(res$statistic[3]))
  expect_true(is.na(res$p.perm[3]))
  expect_true(is.na(res$p.adjusted[3]))
  expect_false("p.perm" %in% names(gpuRowTtest(x, group)))

  expect_error(gpuRowTtest(x, group[-1]), "one entry per column")
  expect_error(gpuRowTtest(x, rep(1:3, length.out = 8)), "two levels")
  expect_error(gpuRowTtest(x, c(1, rep(2, 7))), "at least 2")
  expect_error(gpuRowTtest(x, group, B = -1), "non-negative")
  x[1, 1] <- NA
  expect_error(gpuRowTtest(x, group), "missing")
})

test_that("gpuRowTtest counts a labelling equivalent to the observed one exactly", {
  skip_if_no_gpu()
  # equal groups: a labelling and its swap are the same test. For a row
  # that separates the groups completely, the permutations equal to the
  # observed labelling or its swap are exactly the ones that reach it;
  # the count of those is known from the draws themselves
  set.seed(84)
  n <- 10
  x <- matrix(rnorm(50 * n), 50, n)
  group <- rep(1:2, each = 5)
  x[1, group == 1] <- x[1, group == 1] + 1000
  for (pr in precisions) {
    B <- 3000
    set.seed(85)
    res <- gpuRowTtest(x, group, B = B, precision = pr)
    set.seed(85)
    in1 <- group == 1
    same <- 0
    for (b in seq_len(B)) {
      lab <- in1[sample.int(n)]
      if (all(lab == in1) || all(lab == !in1)) same <- same + 1
    }
    expect_equal(res$p.perm[1], (same + 1) / (B + 1), info = pr)
    expect_equal(res$p.adjusted[1], (same + 1) / (B + 1), info = pr)
  }
})

test_that("gpuRowTtest reports a complete separation as infinite", {
  skip_if_no_gpu()
  set.seed(86)
  x <- matrix(rnorm(4 * 8), 4, 8)
  group <- rep(c("a", "b"), each = 4)
  x[2, ] <- rep(c(1, 2), each = 4)      # constant within each group, apart
  x[3, ] <- 7                            # constant: no test
  B <- 500
  set.seed(87)
  res <- gpuRowTtest(x, group, B = B)
  expect_equal(res$statistic[2], -Inf)
  expect_equal(res$p.value[2], 0)
  expect_true(is.na(res$df[2]))
  # only a relabelling that separates the groups as completely (the
  # observed one or its swap) reaches an infinite t: count those draws
  set.seed(87)
  in1 <- group == "a"
  same <- 0
  for (b in seq_len(B)) {
    lab <- in1[sample.int(8)]
    if (all(lab == in1) || all(lab == !in1)) same <- same + 1
  }
  expect_equal(res$p.perm[2], (same + 1) / (B + 1))
  expect_equal(res$p.adjusted[2], (same + 1) / (B + 1))
  expect_true(is.nan(res$statistic[3]) && is.na(res$p.perm[3]))
  expect_true(all(is.finite(res$statistic[c(1, 4)])))
  # Student's df is defined there
  pooled <- gpuRowTtest(x, group, var.equal = TRUE)
  expect_equal(pooled$statistic[2], -Inf)
  expect_equal(pooled$df[2], 6)
})
