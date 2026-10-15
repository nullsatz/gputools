# Where each gputools function starts to beat base R: times the GPU
# (single and double precision) and the CPU over a range of sizes, and the
# fixed cost of one small call. Writes vignettes/crossover.csv and
# vignettes/call-overhead.csv for the benchmarks vignette.
#
#   Rscript bench/crossover.R [output directory]
#
# Run from the top of the source checkout with gputools installed, on the
# machine the benchmarks vignette describes.

library(gputools)
stopifnot(gpuAvailable())

out <- commandArgs(trailingOnly = TRUE)
out <- if (length(out)) out[1] else "vignettes"

# median of reps timings after one warm-up; small cases repeat inside the
# timing so the clock's resolution does not dominate
timeIt <- function(f, reps = 3L) {
  f()
  once <- system.time(f())[["elapsed"]]
  inner <- max(1L, as.integer(ceiling(0.2 / max(once, 1e-4))))
  median(vapply(seq_len(reps), function(i)
    system.time(for (k in seq_len(inner)) f())[["elapsed"]] / inner,
    numeric(1)))
}

rows <- list()
add <- function(fn, what, size, cpu, single, double) {
  cat(sprintf("%-12s %-28s %8d  cpu %9.5f  single %9.5f  double %9.5f\n",
              fn, what, size, cpu, single, double))
  rows[[length(rows) + 1L]] <<- data.frame(fn = fn, what = what, size = size,
    cpu = cpu, single = single, double = double)
}
sweep <- function(fn, what, sizes, make, cpu, gpu) {
  for (s in sizes) {
    d <- make(s)
    add(fn, what, s, if (is.null(cpu)) NA else timeIt(function() cpu(d)),
        timeIt(function() gpu(d, "single")),
        timeIt(function() gpu(d, "double")))
  }
}

set.seed(1)
m <- function(r, c) matrix(rnorm(r * c), r, c)

sweep("gpuCor", "variables (200 rows)", c(10, 30, 100, 300, 1000, 3000),
      function(p) m(200, p), function(d) cor(d),
      function(d, pr) gpuCor(d, precision = pr))
sweep("gpuCor spearman", "variables (200 rows)", c(10, 30, 100, 300, 1000, 3000),
      function(p) m(200, p), function(d) cor(d, method = "spearman"),
      function(d, pr) gpuCor(d, method = "spearman", precision = pr))
sweep("gpuCor kendall", "variables (200 rows)", c(5, 10, 30, 100, 300),
      function(p) m(200, p), function(d) cor(d, method = "kendall"),
      function(d, pr) gpuCor(d, method = "kendall", precision = pr))
sweep("gpuDist", "points (20 dimensions)", c(50, 100, 300, 1000, 3000),
      function(n) m(n, 20), function(d) dist(d),
      function(d, pr) gpuDist(d, precision = pr))
sweep("gpuHclust", "points", c(50, 100, 300, 1000, 3000),
      function(n) dist(m(n, 20)), function(d) hclust(d),
      function(d, pr) gpuHclust(d, precision = pr))
sweep("gpuDistClust", "points (20 dimensions)", c(50, 100, 300, 1000, 3000),
      function(n) m(n, 20), function(d) hclust(dist(d)),
      function(d, pr) gpuDistClust(d, precision = pr))

grangerByLm <- function(x, lag) {
  n <- nrow(x); rows <- (lag + 1):n
  lags <- function(v) sapply(1:lag, function(k) v[rows - k])
  for (i in seq_len(ncol(x))) for (j in seq_len(ncol(x))) if (i != j) {
    y <- x[rows, j]
    lm.fit(cbind(1, lags(x[, j])), y)
    lm.fit(cbind(1, lags(x[, j]), lags(x[, i])), y)
  }
}
sweep("gpuGranger", "series (300 times, lag 2)", c(5, 10, 20, 50, 100),
      function(p) m(300, p), function(d) grangerByLm(d, 2),
      function(d, pr) gpuGranger(d, lag = 2, precision = pr))

lmData <- function(n) data.frame(y = rnorm(n), m(n, 20))
sweep("gpuLm", "rows (20 predictors)", c(1e3, 1e4, 1e5, 1e6),
      lmData, function(d) lm(y ~ ., d),
      function(d, pr) gpuLm(y ~ ., d, useSingle = pr == "single"))
glmData <- function(n) { d <- lmData(n); d$y <- rbinom(n, 1, 0.5); d }
sweep("gpuGlm", "rows (20 predictors)", c(1e3, 1e4, 1e5),
      glmData, function(d) glm(y ~ ., binomial, d),
      function(d, pr) gpuGlm(y ~ ., binomial, d, useSingle = pr == "single"))
sweep("gpuQr", "columns (2 x as many rows)", c(50, 100, 300, 1000, 2000),
      function(p) m(2 * p, p), function(d) qr(d),
      function(d, pr) gpuQr(d, precision = pr))
sweep("gpuSolve", "n (n x n system)", c(50, 100, 300, 1000, 3000),
      function(n) list(a = crossprod(m(n + 10, n)), b = rnorm(n)),
      function(d) solve(d$a, d$b),
      function(d, pr) gpuSolve(d$a, d$b, precision = pr))
sweep("gpuPrcomp", "variables (1,000 rows)", c(20, 100, 300, 1000),
      function(p) m(1000, p), function(d) prcomp(d),
      function(d, pr) gpuPrcomp(d, precision = pr))
sweep("gpuMatMult", "n (n x n by n x n)", c(100, 300, 1000, 3000),
      function(n) list(a = m(n, n), b = m(n, n)), function(d) d$a %*% d$b,
      function(d, pr) gpuMatMult(d$a, d$b))

res <- do.call(rbind, rows)
dir.create(out, showWarnings = FALSE)
write.csv(res, file.path(out, "crossover.csv"), row.names = FALSE)

# the fixed cost of a call, and what a loop of small calls costs
small <- m(50, 2)
overhead <- data.frame(
  what = c("one gpuCor call, 50 x 2", "one cor call, 50 x 2",
           "1,000 pairs: a loop of gpuCor calls",
           "1,000 pairs: one gpuCor call for all 1,000 x 1,000 (the 1,000 wanted are its diagonal)",
           "1,000 pairs: a loop of cor calls"),
  seconds = c(timeIt(function() gpuCor(small)), timeIt(function() cor(small)),
              NA, NA, NA))
a <- m(50, 1000); b <- m(50, 1000)
overhead$seconds[3] <- timeIt(function()
  for (i in 1:1000) gpuCor(a[, i, drop = FALSE], b[, i, drop = FALSE]), 1L)
overhead$seconds[4] <- timeIt(function() gpuCor(a, b))
overhead$seconds[5] <- timeIt(function()
  for (i in 1:1000) cor(a[, i], b[, i]), 1L)
print(overhead)
write.csv(overhead, file.path(out, "call-overhead.csv"), row.names = FALSE)
