\name{gpuSolve}
\alias{gpuSolve}
\title{Estimate the solution to a matrix vector equation}

\description{
	This function estimates the solution to an equation of the form x * b = y
	where x is a matrix, b is an unknown vector, and y is a known vector.  It
	does much calculation on a GPU. If the y argument is omitted, the
	function returns the inverse of x.

	The function uses R's base 'qr' and then applies the gpu to the result
	to get the final solution.
}

\usage{
	gpuSolve(x, y=NULL)
}

\arguments{
	\item{x}{a matrix of floating point numbers.}  
	\item{y}{a vector of floating point numbers of length nrow(x).}
}

\value{
a vector or matrix of floating point numbers. If y is not null,
then the value is an estimate of the vector b of length ncol(x)
where x * b = y.  If y is null or omitted, the value is a matrix,
an estimate of a matrix multiplicative pseudo inverse of x.
}

\examples{
	x <- matrix(runif(100), 10, 10)
	y <- runif(10)
	b <- gpuSolve(x, y)
	cat("Solution:\n")
	print(b)
	x.inverse <- gpuSolve(x)
	cat("an estimate of a pseudo inverse for x:\n")
	print(x.inverse)
}
