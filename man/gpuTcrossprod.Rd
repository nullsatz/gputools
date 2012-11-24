\name{gpuTcrossprod}
\alias{gpuTcrossprod}
\title{Perform Matrix Transposed Cross-product with a GPU}

\description{
	Performs transposed matrix cross-product using a GPU.  This function is merely a 
	couple of wrappers for the CUBLAS cublasSgemm function.
}

\usage{
	gpuTcrossprod(a, b)
}

\arguments{
	\item{a}{a matrix of floating point values.}
	\item{b}{a matrix of floating point values.  If null, defaultsto
	'a'.}
}

\value{
	A matrix of single precision floating point values.
	The matrix is the transposed cross-product of arguments 'a'
	and 'b', i.e., a * t(b).
}

\examples{
matA <- matrix(runif(2*3), 2, 3)
matB <- matrix(runif(4*3), 4, 3)
gpuTcrossprod(matA, matB)
}

\keyword{array}
\keyword{algebra}
