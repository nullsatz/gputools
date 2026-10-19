\name{gpuGranger}
\alias{gpuGranger}
\title{Granger Causality Tests for Every Pair of Series on a GPU}

\description{
  Tests, for every ordered pair of time series, whether the past of one
  improves the prediction of the other beyond the other's own past: the
  F test of two least-squares fits that \pkg{lmtest}'s
  \code{grangertest} and statsmodels' \code{grangercausalitytests}
  (\code{ssr_ftest}) report. Every pair at once, on the GPU, in double
  precision.
}

\usage{
gpuGranger(x, y=NULL, lag,
    precision = getOption("gputools.precision", "single"))
}

\arguments{
  \item{x}{a numeric matrix, one series per column in time order down the
    rows, with no missing values. Its columns are the causes, and with
    \code{y} missing the effects too.}
  \item{y}{an optional numeric matrix of effects, the same times in the
    same rows.}
  \item{lag}{how many past values of each series the models use: a whole
    number from 1 to 16.}
  \item{precision}{accepted for the common interface, but the Granger
    tests always compute in double: in single precision the F statistics
    of long random walks were off by more than 10 percent.}
}

\details{
  With \eqn{T = n - lag} usable times, the restricted model of effect
  \eqn{j} regresses it on an intercept and its own \eqn{lag} past values,
  the unrestricted one adds cause \eqn{i}'s \eqn{lag} past values, and
  \deqn{F = \frac{(RSS_r - RSS_u) / lag}{RSS_u / (T - 2\,lag - 1)}}
  on \eqn{(lag, T - 2\,lag - 1)} degrees of freedom.

  By the Frisch-Waugh theorem \eqn{RSS_r - RSS_u = g' G^{-1} g} where,
  after the effect's restricted design is projected out of everything
  (through an orthonormal basis of it: Cholesky-QR2, or Householder QR for
  designs too ill-conditioned for it), \eqn{M} holds the cause's lags with
  the effect's own past removed, \eqn{g = M'e} is their product with the
  effect's residual and \eqn{G = M'M}. Each effect's basis is built once,
  and every cause's test against it comes from a few matrix products. For
  most pairs \eqn{G} and \eqn{g} are taken from the small products
  \eqn{L'L - C'C} and \eqn{L'e - C'(Q'e)} with \eqn{C = Q'L}; that form
  subtracts numbers of the size of the cause's lags to leave what is
  beside the effect's own past, so where its rounding error could matter
  (judged pair by pair from the conditioning of \eqn{G}, how much of the
  cause is left, and the size of F) the pair is recomputed with \eqn{M}
  formed explicitly. Each series is first centred and scaled, which
  changes no F. In double precision the F statistics agree with
  \code{lm()} to about 1e-8.

  A test is \code{NA}, with a warning, when it cannot be resolved in
  double precision: when the effect's restricted design (intercept and
  lags, each at unit length) has condition number above 1e10, or a
  cause's \eqn{G} above 1e12, or when less than 1e-20 of a cause's lags
  (in squared length) is left beside the effect's own past, a cause that
  is, to rounding, a linear function of it. (The condition numbers are
  the estimates \eqn{\|A\|_F \|A^{-1}\|_F}, within a factor of the number
  of columns of the usual ones.) White noise, near-unit-root series,
  random walks with or without drift and trends stay far below these
  limits; series integrated two or three times can exceed them at long
  lags, and should be differenced first, as Granger tests assume
  stationary series anyway. Exactly periodic series (a pure sinusoid)
  have collinear lags too.

  A series is not tested against itself (its lags are already in the
  restricted model), and a constant series has no tests: \code{NA}, with
  no warning.
}

\value{a list of two numeric matrices, \code{fStatistics} and
  \code{pValues}, each with a row per column of \code{x} and a column per
  column of \code{y} (of \code{x} when \code{y} is missing), with dimnames
  \code{cause} (rows) and \code{effect} (columns) taken from the column
  names of \code{x} and \code{y}.

  \code{fStatistics[i, j]} answers "does \code{x[, i]} Granger-cause
  \code{y[, j]}?" (or \code{x[, j]}), and \code{pValues[i, j]} is its
  p-value from the F distribution on \eqn{(lag, n - 3\,lag - 1)} degrees
  of freedom. With \code{y} missing the diagonal, a series against itself,
  is \code{NA}.
}

\references{
  Granger, C. W. J. (1969). Investigating causal relations by econometric
  models and cross-spectral methods. \emph{Econometrica} 37(3), 424-438.
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
set.seed(1)
x <- matrix(rnorm(300 * 4), 300, 4)
x[2:300, 3] <- x[2:300, 3] + 0.8 * x[1:299, 1]   # series 1 drives series 3
res <- gpuGranger(x, lag = 2)
which(res$pValues < 1e-6, arr.ind = TRUE)         # cause 1, effect 3

# causes in x, effects in y
y <- matrix(rnorm(300 * 2), 300, 2)
y[2:300, 2] <- y[2:300, 2] + 0.5 * x[1:299, 4]
gpuGranger(x, y, lag = 1)$pValues
\dontshow{\}) # examplesIf}
}
