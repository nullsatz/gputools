\name{gpuMi}
\alias{gpuMi}
\title{B-Spline Mutual Information of Every Pair of Variables on a GPU}

\description{
  Estimates the mutual information between every column of \code{x} and
  every column of \code{y} (of \code{x} itself when \code{y} is missing)
  by the B-spline estimator of Daub et al. (2004), on the GPU. Unlike a
  correlation it detects any kind of dependence, not only linear or
  monotone association.
}

\usage{
gpuMi(x, y = NULL, bins = 10, splineOrder = 3,
    precision = getOption("gputools.precision", "single"))
}

\arguments{
  \item{x}{a numeric matrix, one variable per column, a sample per row,
    with no missing values. The values may lie in any range: each variable
    is scaled onto the spline domain by its own minimum and maximum.}
  \item{y}{an optional numeric matrix of further variables, the same
    samples in the same rows.}
  \item{bins}{the number of bins each variable's range is divided into
    (\eqn{M} in Daub et al., who find 10 to work well), at most 64.}
  \item{splineOrder}{the B-spline order (\eqn{k} in Daub et al.), from 1 to
    \code{bins}: 1 puts each value wholly in one bin (plain binning), 2
    spreads it linearly over two, 3 (which Daub et al. recommend)
    quadratically over three, and so on. The polynomial degree is
    \code{splineOrder - 1}.}
  \item{precision}{\code{"single"} or \code{"double"}: the floating point
    precision of the products on the GPU. The B-spline weights themselves
    are always computed in double from the data as given, since rounding
    a value on a knot (as discrete data put them) to single would move it
    into the neighbouring bin. Results are returned as doubles either way.}
}

\details{
  Each variable is scaled linearly onto the spline domain \eqn{[0, M - k +
  1]}, its minimum at 0 and its maximum at the far end, with the knot
  vector \eqn{0, \ldots, 0, 1, 2, \ldots, M - k, M - k + 1, \ldots, M - k
  + 1} (each end repeated \eqn{k} times), as in Daub et al.; each value
  then has B-spline weights over the \eqn{M} bins, non-negative and
  summing to 1. With \eqn{w_i(s, a)} the weight of sample \eqn{s} of
  variable \eqn{i} in bin \eqn{a}, the bin probabilities are \eqn{P_i(a) =
  \mathrm{mean}_s\, w_i(s, a)}{P_i(a) = mean_s w_i(s, a)} and the joint
  ones \eqn{P_{ij}(a, b) = \mathrm{mean}_s\, w_i(s, a) w_j(s, b)}{P_ij(a,
  b) = mean_s w_i(s, a) w_j(s, b)}, so that \eqn{MI = H_i + H_j - H_{ij}}
  with \eqn{H = -\sum P \log_2 P}{H = -sum P log2 P}.

  On the GPU, the weights of many variables side by side make one matrix
  product hold the joint distribution of every pair, a \eqn{M \times M}{M
  x M} block per pair. Since each sample's weights sum to 1, a block's row
  and column sums are the two marginals, and the information is computed
  from the block alone as \eqn{\sum P_{ij} \log_2 (P_{ij} / (P_i
  P_j))}{sum P_ij log2(P_ij / (P_i P_j))}: the same quantity, without
  subtracting large entropies to get a small difference, and consistent
  with any rounding in the product. \eqn{0 \log 0}{0 log 0} counts as 0,
  and an estimate that rounds just below 0 is 0.
}

\value{a numeric matrix with a row per column of \code{y} (of \code{x}
  when \code{y} is missing) and a column per column of \code{x}: entry
  \eqn{(j, i)} is the mutual information of \code{y[, j]} with
  \code{x[, i]}, in bits. A constant variable shares no information: 0.
  With \code{y} missing, the diagonal is each variable's information with
  itself: its entropy for \code{splineOrder = 1}, and less for higher
  orders, whose spread weights blur a variable's joint distribution with
  itself.
}

\references{
  Daub, C. O., Steuer, R., Selbig, J. and Kloska, S. (2004). Estimating
  mutual information using B-spline functions -- an improved similarity
  measure for analysing gene expression data. \emph{BMC Bioinformatics}
  5, 118. \doi{10.1186/1471-2105-5-118}
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
set.seed(1)
u <- runif(2000, -1, 1)
x <- cbind(u = u, usq = u^2, noise = runif(2000, -1, 1))
mi <- gpuMi(x)
mi["usq", "u"] > 10 * mi["noise", "u"]   # u and u^2: dependent, uncorrelated
abs(cor(u, u^2)) < 0.05
\dontshow{\}) # examplesIf}
}
