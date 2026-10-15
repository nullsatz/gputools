# gpuCorPerm against the same permutation test on the CPU: B = 1000
# permutations, maxT included, written two ways in R -- a loop of cor()
# calls, and the faster loop of BLAS products of standardised columns.
#
#   Rscript bench/perm.R [output directory]

library(gputools)
stopifnot(gpuAvailable())
out <- commandArgs(trailingOnly = TRUE)
out <- if (length(out)) out[1] else "vignettes"

once <- function(f) { gc(); system.time(f())[["elapsed"]] }
B <- 1000
n <- 50
cpuPerm <- function(x, y, perms, product) {
  z <- function(m) { m <- scale(m); m / sqrt(nrow(m) - 1) }
  if (product) { zx <- z(x); zy <- z(y); r0 <- crossprod(zx, zy) }
  else r0 <- cor(x, y)
  counts <- matrix(0L, ncol(x), ncol(y)); maxima <- numeric(B)
  for (b in seq_len(B)) {
    rb <- if (product) crossprod(zx, zy[perms[, b], ]) else cor(x, y[perms[, b], ])
    counts <- counts + (abs(rb) >= abs(r0))
    maxima[b] <- max(abs(rb))
  }
  list(counts, maxima)
}
rows <- list()
for (p in c(50, 200, 1000, 3000)) {
  set.seed(1)
  x <- matrix(rexp(n * p), n, p); y <- matrix(rexp(n * p), n, p)
  perms <- vapply(seq_len(B), function(b) sample.int(n), integer(n))
  for (pr in c("single", "double"))                      # warm up
    invisible(gpuCorPerm(x[, 1:10], y[, 1:10], B = 10, precision = pr))
  gpu <- once(function() gpuCorPerm(x, y, B = B))
  gpuD <- once(function() gpuCorPerm(x, y, B = B, precision = "double"))
  loopCor <- if (p <= 1000) once(function() cpuPerm(x, y, perms, FALSE)) else NA
  loopBlas <- once(function() cpuPerm(x, y, perms, TRUE))
  cat(sprintf("p = q = %5d (%8.0f pairs)  cor loop %8.2f  BLAS loop %8.2f  gpu %6.3f  double %6.3f\n",
              p, p * as.double(p), loopCor, loopBlas, gpu, gpuD))
  rows[[length(rows) + 1]] <- data.frame(variables = p, rows = n,
    permutations = B, corLoop = loopCor, blasLoop = loopBlas, single = gpu,
    double = gpuD)
}
write.csv(do.call(rbind, rows), file.path(out, "perm.csv"), row.names = FALSE)
