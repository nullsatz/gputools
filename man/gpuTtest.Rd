\name{gpuTtest}
\alias{gpuTtest}
\title{T-Test Estimator with a GPU}

\description{
	Given the number of samples and a Pearson correlation coefficient, this 
	function estimates the t-score on a GPU. If an entry in goodPairs is
	zero or one then you may get a NaN as the t-test result.
}

\usage{
	gpuTtest(goodPairs, coeffs)
}

\arguments{
	\item{goodPairs}{a vector of positive integer values.  Value i 
		represents the number of samples used to calculate the i-th 
		value of the 'coeffs' argument.}
	\item{coeffs}{a vector of floating point values representing
		Pearson correlation coefficients.}
}

\value{a vector of single precision floating point values.  The i-th entry
	is an estimate of the t-score of the i-th entry of the 'coeffs' argument.
}

\seealso{
	\code{\link{gpuCor}}.
}

\examples{
goodPairs <- rpois(10, lambda=5)
coeffs <- runif(10)
gpuTtest(goodPairs, coeffs)
}
