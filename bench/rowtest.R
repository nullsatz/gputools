# gpuRowTtest's permutations against the same permutation test written in
# vectorised R (each permutation's Welch t for every row from rowMeans and
# rowSums, maxT included): genes x samples, B = 1000.
#
#   Rscript bench/rowtest.R [output directory]

library(gputools)
stopifnot(gpuAvailable())
out <- commandArgs(trailingOnly = TRUE)
out <- if (length(out)) out[1] else "vignettes"

once <- function(f) { gc(); system.time(f())[["elapsed"]] }
B <- 1000
welch <- function(x, in1) {
  n1 <- sum(in1); n2 <- length(in1) - n1
  a <- x[, in1, drop = FALSE]; b <- x[, !in1, drop = FALSE]
  m1 <- rowMeans(a); m2 <- rowMeans(b)
  v1 <- rowSums((a - m1)^2) / (n1 - 1); v2 <- rowSums((b - m2)^2) / (n2 - 1)
  (m1 - m2) / sqrt(v1 / n1 + v2 / n2)
}
cpuPerm <- function(x, in1) {
  t0 <- welch(x, in1)
  counts <- numeric(nrow(x)); maxima <- numeric(B)
  for (b in seq_len(B)) {
    tb <- abs(welch(x, in1[sample.int(length(in1))]))
    counts <- counts + (tb >= abs(t0))
    maxima[b] <- max(tb)
  }
  list(counts, maxima)
}
rows <- list()
for (sz in list(c(1000, 20), c(20000, 20), c(20000, 100), c(50000, 100))) {
  genes <- sz[1]; n <- sz[2]
  set.seed(1)
  x <- matrix(rnorm(genes * n), genes, n)
  group <- rep(1:2, length.out = n)
  invisible(gpuRowTtest(x[1:10, ], group, B = 10))      # warm up
  noPerm <- once(function() gpuRowTtest(x, group))
  gpu <- once(function() gpuRowTtest(x, group, B = B))
  gpuD <- once(function() gpuRowTtest(x, group, B = B, precision = "double"))
  cpu <- once(function() cpuPerm(x, group == 1))
  cat(sprintf("genes %6d samples %4d  R loop %7.2f  gpu %6.3f  double %6.3f  (no permutations %6.3f)\n",
              genes, n, cpu, gpu, gpuD, noPerm))
  rows[[length(rows) + 1]] <- data.frame(genes = genes, samples = n,
    permutations = B, cpu = cpu, single = gpu, double = gpuD,
    noPermutations = noPerm)
}
write.csv(do.call(rbind, rows), file.path(out, "rowtest.csv"),
          row.names = FALSE)
