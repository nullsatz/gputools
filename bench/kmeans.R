# gpuKmeans against kmeans(): the same random starts, 20 iterations at
# most, over numbers of points and of clusters. Base R is timed with
# Lloyd's algorithm (the one gpuKmeans runs) and with its default,
# Hartigan-Wong; their total within-cluster sums of squares are recorded
# too, since Hartigan-Wong usually ends lower from the same start.
#
#   Rscript bench/kmeans.R [output directory]

library(gputools)
stopifnot(gpuAvailable())
out <- commandArgs(trailingOnly = TRUE)
out <- if (length(out)) out[1] else "vignettes"

once <- function(f) { gc(); system.time(f())[["elapsed"]] }
d <- 50
rows <- list()
for (sz in list(c(1e3, 10), c(1e4, 10), c(1e4, 100), c(1e5, 10),
                c(1e5, 100), c(1e5, 1000), c(1e6, 100), c(1e6, 1000))) {
  n <- sz[1]; k <- sz[2]
  set.seed(1)
  mids <- matrix(rnorm(k * d, sd = 3), k, d)
  x <- mids[sample(k, n, replace = TRUE), ] + matrix(rnorm(n * d), n, d)
  starts <- x[sample(n, k), ]
  invisible(gpuKmeans(x[1:200, ], 5))                  # warm up
  run <- function(f) {
    res <- NULL
    t <- once(function() res <<- suppressWarnings(f()))
    c(t, res$tot.withinss, res$iter)
  }
  g <- run(function() gpuKmeans(x, starts, iter.max = 20))
  gd <- run(function() gpuKmeans(x, starts, iter.max = 20,
                                 precision = "double"))
  # beyond 1e8 point-centre pairs base R takes many minutes a run
  cpu <- n * as.double(k) <= 1e8
  l <- if (cpu) run(function()
    kmeans(x, starts, iter.max = 20, algorithm = "Lloyd")) else rep(NA, 3)
  h <- if (cpu) run(function() kmeans(x, starts, iter.max = 20))
    else rep(NA, 3)
  cat(sprintf("n %7d k %5d  Lloyd %8.2f (iter %s)  H-W %8.2f  gpu %7.3f  double %7.3f  wss gpu/H-W %.4f\n",
              n, k, l[1], format(l[3]), h[1], g[1], gd[1], g[2] / h[2]))
  rows[[length(rows) + 1]] <- data.frame(points = n, clusters = k,
    dimensions = d, lloyd = l[1], hartiganWong = h[1], single = g[1],
    double = gd[1], iterations = g[3], wssLloyd = l[2], wssHW = h[2],
    wssGpu = g[2])
}
write.csv(do.call(rbind, rows), file.path(out, "kmeans.csv"),
          row.names = FALSE)
