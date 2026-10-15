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

# Agglomerative clustering by the Lance-Williams (1967) update,
#   d(k, i+j) = ai d(k,i) + aj d(k,j) + b d(i,j) + g |d(k,i) - d(k,j)|,
# straight from the published coefficients; returns the merge heights.
# flexible: Lance & Williams' flexible beta, ai = aj = (1 - lambda) / 2;
# flexible group: Belbin, Faith & Milligan (1992), ai = (1 - lambda) ni / n.
lanceWilliams <- function(d, method, lambda = 0.5, beta = 0.5) {
  d <- as.matrix(d)
  n <- nrow(d)
  size <- rep(1, n)
  active <- rep(TRUE, n)
  diag(d) <- Inf
  heights <- numeric(n - 1)
  for (step in seq_len(n - 1)) {
    dd <- d
    dd[!active, ] <- Inf
    dd[, !active] <- Inf
    w <- which(dd == min(dd), arr.ind = TRUE)[1, ]
    i <- min(w); j <- max(w)
    dij <- d[i, j]
    heights[step] <- dij
    ni <- size[i]; nj <- size[j]
    for (k in which(active)) {
      if (k == i || k == j) next
      nk <- size[k]; dki <- d[k, i]; dkj <- d[k, j]
      co <- switch(method,
        single = c(1/2, 1/2, 0, -1/2),
        complete = c(1/2, 1/2, 0, 1/2),
        average = c(ni, nj, 0, 0) / c(ni + nj, ni + nj, 1, 1),
        mcquitty = , wpgma = c(1/2, 1/2, 0, 0),
        median = c(1/2, 1/2, -1/4, 0),
        centroid = c(ni / (ni + nj), nj / (ni + nj),
                     -ni * nj / (ni + nj)^2, 0),
        ward = c((ni + nk), (nj + nk), -nk, 0) / (ni + nj + nk),
        flexible = c((1 - lambda) / 2, (1 - lambda) / 2, beta, 0),
        "flexible group" = c((1 - lambda) * ni / (ni + nj),
                             (1 - lambda) * nj / (ni + nj), beta, 0))
      v <- co[1] * dki + co[2] * dkj + co[3] * dij + co[4] * abs(dki - dkj)
      d[k, i] <- d[i, k] <- v
    }
    size[i] <- ni + nj
    active[j] <- FALSE
    d[j, ] <- d[, j] <- Inf
  }
  heights
}

# Mutual information by Daub et al. (2004), "Estimating mutual information
# using B-spline functions", BMC Bioinformatics 5:118, from the paper:
# each variable is scaled linearly onto the spline domain [0, M - k + 1],
# each value is spread over the M bins with weights B_{i,k}(z) (Cox-de Boor
# on the knot vector 0..0, 1, 2, ..., M-k+1..M-k+1), probabilities are the
# mean weights, entropies are in bits, and MI = H(x) + H(y) - H(x, y).
bsplineWeights <- function(x, bins, order) {
  knots <- c(rep(0, order), seq_len(bins - order),
             rep(bins - order + 1, order))
  z <- (x - min(x)) / (max(x) - min(x)) * (bins - order + 1)
  basis <- function(i, k, z) {        # B_{i,k}, i = 1..bins, Cox-de Boor
    if (k == 1) {
      inside <- knots[i] <= z & z < knots[i + 1]
      # the right end of the domain belongs to the last non-empty interval
      last <- z == knots[length(knots)] & knots[i + 1] == knots[length(knots)] &
        knots[i] < knots[i + 1]
      return(as.numeric(inside | last))
    }
    left <- if (knots[i + k - 1] > knots[i])
      (z - knots[i]) / (knots[i + k - 1] - knots[i]) * basis(i, k - 1, z) else 0
    right <- if (knots[i + k] > knots[i + 1])
      (knots[i + k] - z) / (knots[i + k] - knots[i + 1]) *
        basis(i + 1, k - 1, z) else 0
    left + right
  }
  sapply(seq_len(bins), function(i) basis(i, order, z))
}
entropyBits <- function(p) -sum(p[p > 0] * log2(p[p > 0]))
daubMI <- function(x, y, bins, order) {
  wx <- bsplineWeights(x, bins, order)
  wy <- bsplineWeights(y, bins, order)
  entropyBits(colMeans(wx)) + entropyBits(colMeans(wy)) -
    entropyBits(crossprod(wx, wy) / length(x))
}
