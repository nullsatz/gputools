# Single-precision results get a float32-sized tolerance; double-precision
# ones should agree with base R almost to the last digit.
tol <- 1e-4
precisions <- c("single", "double")
tolFor <- function(precision) if (precision == "double") 1e-10 else tol

skip_if_no_gpu <- function() {
  if (!gpuAvailable()) skip("no usable CUDA device")
}

# Granger causality F-test, "does cause[] Granger-cause effect[]", by lm().
grangerRef <- function(effect, cause, lag) {
  n <- length(effect)
  rows <- (lag + 1):n
  lags <- function(v) sapply(1:lag, function(k) v[rows - k])
  y <- effect[rows]
  restricted <- lm(y ~ lags(effect))
  unrestricted <- lm(y ~ lags(effect) + lags(cause))
  rssR <- sum(residuals(restricted)^2)
  rssU <- sum(residuals(unrestricted)^2)
  df2 <- length(rows) - 2 * lag - 1
  f <- ((rssR - rssU) / lag) / (rssU / df2)
  c(f = f, p = pf(f, lag, df2, lower.tail = FALSE))
}
