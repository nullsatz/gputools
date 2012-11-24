\name{gpuQr}
\alias{gpuQr}
\title{Estimate the QR decomposition for a matrix}

\description{
	gpuQR estimates the QR decomposition for a matrix using
	column pivoting and householder matrices.  The work is done on a
	GPU.

	Note:  a rank-revealing pivoting scheme is employed, potentially
	resulting in pivot distinctly different from ordinary "qr".
}

\usage{
gpuQr(x, tol = 1e-07)
}

\arguments{
	\item{x}{a matrix of floating point numbers.
		This is the matrix that will	be decomposed into Q and R
		factors.}
	\item{tol}{a floating point value. It is used for estimating the rank
		of matrix x.}
}

\value{
an object of class 'qr'. This object has members qr, qraux, pivot, rank.
It is meant to be identical to the output of R's base function 'qr'.
From the documentation for R's 'qr' function:
The attribute qr is a matrix with the same dimension as 'x'.
The upper triangle contains the R of the QR decomposition.
The lower triangle contains partial information to construct Q.
The attribute qraux is a vector of length 'ncol(x)' contains
more information to construct Q.
The attribute rank is a single integer representing an estimation
of the rank of input matrix x based on the results of the
QR decomposition.  In some cases, this rank can be wildly different from
the actual rank of the matrix x and so is only an estimation.
The attribute pivot contains the permutation applied to columns of x
in the process of calculating the QR decomposition.
}
\author{
The low--level implementation of this function for Nvidia's
CUDA--supporting GPGPUs was written
by Mark Seligman at Rapid Biologics LLC.  
	\code{http://www.rapidbiologics.com}
  }

\references{
  Bischof, C. B. and Van Loan, C. F. (1987)
  The WY Representation for Products of Householder Matrices
  \emph{SIAM J Sci. and Stat. Comp}, \bold{8}, s2--s13.

Bjorck, Ake (1996) \emph{Numerical methods for least squares
  problems}. SIAM.

Golub, Gene H. and Van Loan, C. F. (1996)
 \emph{Matrix Computations}, \bold{Ed. 3}, ch. 5.
}

\examples{
# get some random data of any shape at all
x <- matrix(runif(25), 5, 5)
qr <- gpuQr(x)
print(qr)
}