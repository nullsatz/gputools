\name{gpuSvd}
\alias{gpuSvd}
\title{Singular Value Decomposition on a GPU}
\description{
  The singular value decomposition of a matrix, as \code{\link{svd}}
  computes it, on the GPU with NVIDIA's cuSOLVER (\code{gesvd}).
}
\usage{
gpuSvd(x, nu = min(n, p), nv = min(n, p),
	precision = getOption("gputools.precision", "single"))
}
\arguments{
  \item{x}{a numeric matrix, without missing or infinite values.}
  \item{nu, nv}{the number of left and right singular vectors to return,
    as in \code{svd}; at most \code{min(nrow(x), ncol(x))}, since the thin
    decomposition is computed.}
  \item{precision}{\code{"single"} or \code{"double"}: the floating point
    precision the GPU computes in. Single precision is the default, set with
    \code{options(gputools.precision = "double")}; it runs many times faster
    on most consumer GPUs. Results are returned as doubles either way.}
}
\value{
  As \code{svd}: a list with \code{d}, the singular values in decreasing
  order, and, when asked for, \code{u} and \code{v}, the left and right
  singular vectors as columns. As with any SVD, each pair of singular
  vectors is determined only up to sign, so their signs can differ from
  \code{svd}'s.
}
\seealso{\code{\link{svd}}, \code{\link{gpuPrcomp}}}
\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
x <- matrix(rnorm(200 * 20), 200, 20)
s <- gpuSvd(x, precision = "double")
s$d
max(abs(x - s$u \%*\% diag(s$d) \%*\% t(s$v)))
\dontshow{\}) # examplesIf}
}
\keyword{algebra}
