\name{gpuPrcomp}
\alias{gpuPrcomp}
\title{Principal Components Analysis on a GPU}
\description{
  Principal components of a data matrix, as \code{\link{prcomp}} computes
  them, with the singular value decomposition on the GPU
  (\code{\link{gpuSvd}}).
}
\usage{
gpuPrcomp(x, retx = TRUE, center = TRUE, scale. = FALSE, tol = NULL,
	rank. = NULL, precision = getOption("gputools.precision", "single"))
}
\arguments{
  \item{x, retx, center, scale., tol, rank.}{as for \code{prcomp}.}
  \item{precision}{\code{"single"} or \code{"double"}: the floating point
    precision the GPU computes in. Single precision is the default, set with
    \code{options(gputools.precision = "double")}; it runs many times faster
    on most consumer GPUs. Results are returned as doubles either way.}
}
\value{
  An object of class \code{"prcomp"}, as \code{prcomp} returns, so
  \code{summary}, \code{predict}, \code{biplot} and \code{screeplot} work on
  it. Each component's sign is arbitrary, as with any principal components,
  so a column of \code{rotation} and of \code{x} can be the negative of
  \code{prcomp}'s.
}
\seealso{\code{\link{prcomp}}, \code{\link{gpuSvd}}}
\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
pc <- gpuPrcomp(USArrests, scale. = TRUE)
summary(pc)
head(predict(pc, USArrests), 3)
\dontshow{\}) # examplesIf}
}
\keyword{multivariate}
