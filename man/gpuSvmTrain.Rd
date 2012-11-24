\name{gpuSvmTrain}
\alias{gpuSvmTrain}
\title{Train a support vector machine on a data set}

\description{
	This function trains, with the aid of a GPU, a support vector machine
	using the input data x separated into classes y.  The function is
	capable of both regression (the entries of y are continuous) and
	non-regression (each entry of y is either -1.f or 1.f).  The underlying
	code is adapted from Austin Carpenter's cuSVM which can be found at
	\url{http://patternsonascreen.net/cuSVM.html}
}

\usage{
	gpuSvmTrain(y, x, C = 10, kernelWidth = 0.125, eps = 0.5,
	    stoppingCrit = 0.001, isRegression = FALSE)
}

\arguments{
	\item{y}{a vector of floating point numbers.  The length of y should equal
		the number of rows of x.  In the case of isRegression = FALSE, each 
		entry of y is the category of the row of data x.  The negative category
		is indicated by -1 and the positive category is indicated by 1.  In the
		case of isRegression = TRUE, the values of y may take any value between
		-1 and 1 inclusive. These categories are used to train the svm.}
	\item{x}{a matrix of floating point numbers.  Each row i is a point with a
		category given by y[i]. This is the data set used for training the svm.}
	\item{C}{a single floating point number.  This is the SVM regularization
		parameter.}
	\item{kernelWidth}{a single floating point number.  This is the scalar 
		Gaussian kernel parameter.}
	\item{eps}{a single floating point number.  This is the epsilon used in
		regression mode.}
	\item{stoppingCrit}{a single floating point number.  This is the
		optimization stopping criterion.}
	\item{isRegression}{a single logical value.  If isRegression is set to
		TRUE then regression is performed and the y value may be continuously
		valued.  If not, then we use normal svm training and each value in y
		must be either -1 or 1.}
}

\value{
a list consisting of the following elements: supportVectors, svCoefficients,
and svOffset.  The element supportVectors is a matrix of single precision
floating point numbers.  These are the support vectors corresponding to the
coefficients in svCoefficients.  Row i of supportVectors contains ncol(x)
columns and has coefficient svCoefficients[i].  The element svCoefficients is
a single precision vector of the support vector coefficients.  The element
svOffset is a single floating point number of single precision.  It is the
offset for the prediction function.
}

\references{
	Carpenter, Austin, \emph{cuSVM: a cuda implementation of support vector
		classification and regression},
		\url{http://http://patternsonascreen.net/cuSVM.html}
}

\examples{
# y is discrete: -1 or 1 and we set isRegression to FALSE
y <- round(runif(100, min = 0, max = 1))
for(i in 1:5) { if(y[i] == 0) {y[i] <- -1}}

x <- matrix(runif(500), 100, 5)

# a <- gpuSvmTrain(y, x, isRegression = FALSE)
# print(a)

# b <- gpuSvmPredict(x, a$supportVectors, a$svCoefficients, a$svOffset, isRegression = FALSE)
# print(b)

# this time around, y : -1 or 1 and we set isRegression to FALSE
y <- runif(100, min = -1, max = 1)

x <- matrix(runif(500), 100, 5)

# a <- gpuSvmTrain(y, x, isRegression = TRUE)
# print(a)

# b <- gpuSvmPredict(x, a$supportVectors, a$svCoefficients, a$svOffset, isRegression = TRUE)
# print(b)
}
