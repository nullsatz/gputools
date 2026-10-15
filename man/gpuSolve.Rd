\name{gpuSolve}
\alias{gpuSolve}
\title{Estimate the solution to a matrix vector equation}

\description{
	Solves x b = y for b on the GPU, like \code{\link{solve}}. If y is
	omitted, returns the inverse of x.

	A square x is solved by LU decomposition with partial pivoting, as
	\code{solve} does (NVIDIA's cuSOLVER, \code{getrf}/\code{getrs}).
	An x with more rows than columns is solved in the least-squares sense
	by Householder QR (\code{geqrf}/\code{ormqr}), and its "inverse" is
	the pseudo-inverse.
}

\usage{
	gpuSolve(x, y=NULL,
		precision = getOption("gputools.precision", "single"))
}

\arguments{
	\item{x}{a matrix of floating point numbers.}  
	\item{y}{a vector of length nrow(x), or a matrix with nrow(x) rows.}
	\item{precision}{\code{"single"} or \code{"double"}: the floating point precision
		the GPU computes in. Single precision is the default, set with
		\code{options(gputools.precision = "double")}; it runs many times
		faster on most consumer GPUs. Results are returned as doubles either way.}
}

\value{
b: a vector of length ncol(x) if y is a vector, otherwise a matrix
with ncol(x) rows. If y is omitted, the ncol(x) by nrow(x) inverse (or
pseudo-inverse) of x. Rows are named by the columns of x, as with
\code{solve}.

A square x that is exactly singular is an error, as with \code{solve};
so is an overdetermined x whose columns are linearly dependent, since
the least-squares solution is then not unique.
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
	x <- matrix(runif(100), 10, 10)
	y <- runif(10)
	b <- gpuSolve(x, y)
	cat("Solution:\n")
	print(b)
	x.inverse <- gpuSolve(x)
	cat("an estimate of a pseudo inverse for x:\n")
	print(x.inverse)
\dontshow{\}) # examplesIf}
}
