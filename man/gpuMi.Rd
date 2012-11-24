\name{gpuMi}
\alias{gpuMi}
\title{B spline based mutual information}

\description{
	This function estimates the mutual information for permutations
	of pairs of columns of a matrix using a B spline approach on a
	GPU device.  Please note, the data must be values from the interval
	[0.0, 1.0].
}

\usage{
gpuMi(x, y = NULL, bins = 2, splineOrder = 1)
}

\arguments{
	\item{x}{a matrix of floating point numbers from the interval [0.0, 1.0].
		Each column represents a list of samples of a random variable.  The
		mutual information between each column of x and each column of y will
		be computed.  If y is NULL then each pair of columns of x will be
		compared.}
	\item{y}{a matrix of floating point numbers from the interval [0.0, 1.0].
		Each column represents a list of samples of a random variable.  The
		mutual information between each column of x and each column of y will
		be computed.  If y is NULL then each pair of columns of x will be
		compared.}
	\item{bins}{a single integer value representing the number of equal
		intervals that [0.0, 1.0] will be divided into in order to determine
		the bins in which to place each value of the columns of x and y.  In
		the case of splineOrder = 1, this determines the histogram for
		traditional mutual information.  For splineOrder > 1, a single value
		may be placed in multiple adjoining bins with varying weights on
		membership.}
	\item{splineOrder}{a single integer value giving the degree of the spline
		polynomials used to define both the number of bins a single value will
		be placed in and the weight of membership given to the value.}
}

\value{
	a matrix of single precision floating point values of order ncol(y) by
	ncol(x).  Entry $(i, j)$ of this matrix represents the mutual information
	calculation for $(y_i, x_j)$.
}

\references{
	Carten O. Daub, Ralf Steuer, Joachim Selbig, and Sebastian Kloska.
	2004.  Estimating mutual information using B-spline functions -- an
	improved similarity measure for analysing gene expression data.
	\emph{BMC Bioinformatics}. 5:118.  Available from
	\url{http://www.biomedcentral.com/1471-2105/5/118}
}

\examples{
	# get 3 random variables each with 20 samples
	x <- matrix(runif(60), 20, 3)
	y <- matrix(runif(60), 20, 3)
	# do something interesting
	y[,2] <- 3.0 * (x[,1] + x[,3])
	z <- gpuMi(x, y, bins = 10, splineOrder = 3)
	print(z)
}
