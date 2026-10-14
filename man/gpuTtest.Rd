\name{gpuTtest}
\alias{gpuTtest}
\title{T-Test Estimator with a GPU}

\description{
	Given the number of samples and a Pearson correlation coefficient, this 
	function estimates the t-score on a GPU. If an entry in goodPairs is
	zero or one then you may get a NaN as the t-test result.
}

\usage{
	gpuTtest(goodPairs, coeffs,
		precision = getOption("gputools.precision", "single"))
}

\arguments{
	\item{goodPairs}{a vector of positive integer values.  Value i 
		represents the number of samples used to calculate the i-th 
		value of the 'coeffs' argument.}
	\item{coeffs}{a vector of floating point values representing
		Pearson correlation coefficients.}
	\item{precision}{\code{"single"} or \code{"double"}: the floating point precision
		the GPU computes in. Single precision is the default, set with
		\code{options(gputools.precision = "double")}; it runs many times
		faster on most consumer GPUs. Results are returned as doubles either way.}
}

\value{a vector of single precision floating point values.  The i-th entry
	is an estimate of the t-score of the i-th entry of the 'coeffs' argument.
}

\seealso{
	\code{\link{gpuCor}}.
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
goodPairs <- rpois(10, lambda=5)
coeffs <- runif(10)
gpuTtest(goodPairs, coeffs)
\dontshow{\}) # examplesIf}
}
