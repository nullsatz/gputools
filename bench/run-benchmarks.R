# Times each gputools function against its base-R counterpart and writes
# vignettes/benchmarks.csv plus vignettes/benchmarks-machine.csv for the
# vignette, which cannot run GPU code when the package is checked.
#
#   Rscript bench/run-benchmarks.R [output directory]
#
# Run it on the machine to be reported, from the top of the checkout,
# with gputools installed. Each timing is the median of `reps` runs after
# one warm-up run; the warm-up also compiles the kernels, whose one-off
# cost is reported separately.

library(gputools)
stopifnot(gpuAvailable())

out <- commandArgs(trailingOnly = TRUE)
out <- if (length(out)) out[1] else "vignettes"
reps <- 3L

elapsed <- function(expr) system.time(expr)[["elapsed"]]
timeIt <- function(f) {
  f()                                   # warm-up (and kernel compile)
  median(vapply(seq_len(reps), function(i) elapsed(f()), numeric(1)))
}

results <- list()
add <- function(task, size, cpu, single, double) {
  cat(sprintf("%-28s %-22s cpu %8.3f  single %8.3f  double %8.3f\n",
              task, size, cpu, single, double))
  results[[length(results) + 1L]] <<- data.frame(
    task = task, size = size, cpu = cpu, single = single, double = double)
}
bench <- function(task, size, cpu, gpu) {
  add(task, size, timeIt(cpu),
      timeIt(function() gpu("single")), timeIt(function() gpu("double")))
}

set.seed(1)

# one-off kernel compilation, in a fresh process per precision
compileCost <- function(precision) {
  cmd <- sprintf(paste0("library(gputools); x <- matrix(runif(200), 20);",
                        "cat(system.time(gpuDist(x, precision = '%s'))",
                        "[['elapsed']])"), precision)
  as.numeric(system2(file.path(R.home("bin"), "Rscript"),
                     c("-e", shQuote(cmd)), stdout = TRUE))
}

for (n in c(2000, 4000)) {
  a <- matrix(rnorm(n * n), n); b <- matrix(rnorm(n * n), n)
  add("gpuMatMult vs %*%", sprintf("%d x %d", n, n),
      timeIt(function() a %*% b),
      NA, timeIt(function() gpuMatMult(a, b)))
}

x <- matrix(rnorm(2000 * 1000), 2000, 1000)
bench("gpuCor pearson vs cor", "2000 obs x 1000 vars",
      function() cor(x), function(p) gpuCor(x, precision = p))

xk <- matrix(rnorm(1000 * 100), 1000, 100)
bench("gpuCor kendall vs cor", "1000 obs x 100 vars",
      function() cor(xk, method = "kendall"),
      function(p) gpuCor(xk, method = "kendall", precision = p))

pts <- matrix(rnorm(5000 * 50), 5000, 50)
bench("gpuDist vs dist", "5000 points x 50 dims",
      function() dist(pts), function(p) gpuDist(pts, precision = p))

d <- dist(pts)
bench("gpuHclust vs hclust", "5000 points",
      function() hclust(d), function(p) gpuHclust(d, precision = p))

bench("gpuDistClust vs hclust(dist())", "5000 points x 50 dims",
      function() hclust(dist(pts)),
      function(p) gpuDistClust(pts, precision = p))

ts <- matrix(rnorm(500 * 50), 500, 50)
grangerByLm <- function(x, lag) {
  n <- nrow(x); rows <- (lag + 1):n
  lags <- function(v) sapply(1:lag, function(k) v[rows - k])
  f <- matrix(0, ncol(x), ncol(x))
  for (i in seq_len(ncol(x))) for (j in seq_len(ncol(x))) if (i != j) {
    y <- x[rows, j]
    r <- lm.fit(cbind(1, lags(x[, j])), y)
    u <- lm.fit(cbind(1, lags(x[, j]), lags(x[, i])), y)
    rssR <- sum(r$residuals^2); rssU <- sum(u$residuals^2)
    f[i, j] <- ((rssR - rssU) / lag) / (rssU / (length(rows) - 2 * lag - 1))
  }
  f
}
bench("gpuGranger vs lm.fit loop", "500 times x 50 series, lag 2",
      function() grangerByLm(ts, 2), function(p) gpuGranger(ts, lag = 2,
                                                             precision = p))

mi <- matrix(rnorm(2000 * 200), 2000, 200)
add("gpuMi (no base-R equivalent)", "2000 obs x 200 vars, 10 bins",
    NA, timeIt(function() gpuMi(mi, bins = 10, splineOrder = 3,
                                precision = "single")),
    timeIt(function() gpuMi(mi, bins = 10, splineOrder = 3,
                            precision = "double")))

dat <- data.frame(y = rnorm(1e5), matrix(rnorm(1e5 * 50), 1e5, 50))
bench("gpuLm vs lm", "100000 obs x 50 predictors",
      function() lm(y ~ ., dat),
      function(p) gpuLm(y ~ ., dat, useSingle = p == "single"))

xq <- matrix(rnorm(4000 * 1000), 4000, 1000)
bench("gpuQr vs qr", "4000 x 1000",
      function() qr(xq), function(p) gpuQr(xq, precision = p))

s <- crossprod(matrix(rnorm(3000 * 2000), 3000, 2000))
bench("gpuSolve vs solve", "2000 x 2000",
      function() solve(s), function(p) gpuSolve(s, precision = p))

add("kernel compile, first use of a file", "gpuDist kernels",
    NA, compileCost("single"), compileCost("double"))

res <- do.call(rbind, results)
dir.create(out, showWarnings = FALSE)
write.csv(res, file.path(out, "benchmarks.csv"), row.names = FALSE)

gpu <- system2("nvidia-smi", c("--query-gpu=name,driver_version",
                               "--format=csv,noheader"), stdout = TRUE)[1]
cpu <- sub(".*: ", "", grep("^model name", readLines("/proc/cpuinfo"),
                            value = TRUE)[1])
blas <- sub(".*/", "", extSoftVersion()[["BLAS"]])
machine <- data.frame(
  item = c("GPU, driver", "CPU", "CPU threads", "BLAS", "R", "gputools",
           "date"),
  value = c(gpu, cpu, Sys.getenv("OMP_NUM_THREADS", "unset"),
            if (nzchar(blas)) blas else La_library(), R.version.string,
            paste0(packageVersion("gputools"),
                   tryCatch(paste0(" (commit ", system2("git", c("rev-parse",
                     "--short", "HEAD"), stdout = TRUE), ")"),
                     error = function(e) "", warning = function(w) "")),
            format(Sys.Date())))
write.csv(machine, file.path(out, "benchmarks-machine.csv"), row.names = FALSE)
print(machine)
