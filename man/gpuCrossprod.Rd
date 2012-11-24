\name{gpuCrossprod}
\alias{gpuCrossprod}
\title{Perform Matrix Cross-product with a GPU}

\description{
	Performs matrix cross-product using a GPU.  This function is merely a 
	couple of wrappers for the CUBLAS cublasSgemm function.
}

\usage{
	gpuCrossprod(a, b=NULL)
}

\arguments{
	\item{a}{a matrix of floating point values.}
	\item{b}{a matrix of floating point values.  A null value
	defaults to 'a'.}
}

\value{
	A matrix of single precision floating point values.
	The matrix is the cross-product of arguments 'a' and 'b', i.e.,
	t(a) * b.
}

\examples{
matA <- matrix(runif(3*2), 3, 2)
matB <- matrix(runif(3*4), 3, 4)
gpuCrossprod(matA, matB)
}

\keyword{array}
\keyword{algebra}

