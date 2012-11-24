\name{gpuMatMult}
\alias{gpuMatMult}
\title{Perform Matrix Multiplication with a GPU}

\description{
	Performs matrix multiplication using a GPU.  This function is merely a 
	couple of wrappers for the CUBLAS cublasSgemm function.
}

\usage{
	gpuMatMult(a, b)
}

\arguments{
	\item{a}{a matrix of floating point values.}
	\item{b}{a matrix of floating point values.}
}

\value{
	A matrix of single precision floating point values.
	The matrix is just the product of arguments 'a' and 'b'.
}

\examples{
matA <- matrix(runif(2*3), 2, 3)
matB <- matrix(runif(3*4), 3, 4)
gpuMatMult(matA, matB)
}

\keyword{array}
\keyword{algebra}
