# gpuCorPairs at genome scale: the time to find every pair of variables
# correlated at |r| >= 0.8, and each variable's 10 strongest partners,
# against base R's way (the whole cor() matrix, then which()) and against
# gpuCor (the whole matrix on the GPU, then which()). Base R stops at the
# sizes where its matrix gets too big to be patient with.
#
#   Rscript bench/pairs.R [output directory]

library(gputools)
stopifnot(gpuAvailable())
out <- commandArgs(trailingOnly = TRUE)
out <- if (length(out)) out[1] else "vignettes"

once <- function(f) { gc(); system.time(f())[["elapsed"]] }
n <- 100
rows <- list()
for (p in c(2000, 5000, 10000, 20000, 50000)) {
  set.seed(1)
  x <- matrix(rnorm(n * p), n, p)
  # a few hundred real modules among the noise
  for (m in seq_len(p / 100)) {
    cols <- sample(p, 5)
    x[, cols] <- x[, cols[1]] + matrix(rnorm(n * 5, sd = 0.5), n, 5)
  }
  invisible(gpuCorPairs(x[, 1:100], threshold = 0.8))   # warm up
  invisible(gpuCor(x[, 1:100]))
  edges <- NA
  gpuPairs <- once(function() edges <<- nrow(gpuCorPairs(x, threshold = 0.8)))
  gpuTop <- once(function() gpuCorPairs(x, top = 10))
  pick <- function(r) { r[lower.tri(r, diag = TRUE)] <- 0
                        which(abs(r) >= 0.8, arr.ind = TRUE) }
  gpuFull <- if (p <= 20000) once(function() pick(gpuCor(x)$coefficients)) else NA
  cpu <- if (p <= 10000) once(function() pick(cor(x))) else NA
  cat(sprintf("p %6d  edges %8d  cor+which %8.2f  gpuCor+which %8.2f  gpuCorPairs %7.2f  top10 %7.2f\n",
              p, edges, cpu, gpuFull, gpuPairs, gpuTop))
  rows[[length(rows) + 1]] <- data.frame(variables = p, rows = n,
    pairs = edges, cor = cpu, gpuCor = gpuFull, gpuCorPairs = gpuPairs,
    top10 = gpuTop)
}
res <- do.call(rbind, rows)
write.csv(res, file.path(out, "pairs.csv"), row.names = FALSE)
