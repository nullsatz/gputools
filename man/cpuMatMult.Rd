\name{cpuMatMult}
\alias{cpuMatMult}
\title{Perform Matrix Multiplication}

\description{
	Performs matrix multiplication using R's BLAS.  This function is merely a 
	wrapper for the BLAS dgemm function.
}

\usage{
	cpuMatMult(a, b)
}

\arguments{
	\item{a}{a numeric matrix.}
	\item{b}{a numeric matrix.}
}

\value{
	A numeric matrix.
	The matrix is just the product of arguments 'a' and 'b'.
}

\examples{
matA <- matrix(runif(2*3), 2, 3)
matB <- matrix(runif(3*4), 3, 4)
cpuMatMult(matA, matB)
}

\keyword{array}
\keyword{algebra}
