# Results come back in single precision, so compare with a float32-sized
# tolerance rather than R's double-precision default.
tol <- 1e-4

skip_if_no_gpu <- function() {
  ok <- tryCatch({ getGpuId(); TRUE }, error = function(e) FALSE)
  if (!ok) skip("no usable CUDA device")
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
