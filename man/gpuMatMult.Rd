\name{gpuMatMult}
\alias{gpuMatMult}
\title{Perform Matrix Multiplication with a GPU}

\description{
	Performs matrix multiplication using a GPU.  This function is merely a 
	wrapper for the CUBLAS cublasDgemm function.
}

\usage{
	gpuMatMult(a, b)
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
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
matA <- matrix(runif(2*3), 2, 3)
matB <- matrix(runif(3*4), 3, 4)
gpuMatMult(matA, matB)
\dontshow{\}) # examplesIf}
}

\keyword{array}
\keyword{algebra}
