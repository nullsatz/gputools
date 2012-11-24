\name{gpuSvmPredict}
\alias{gpuSvmPredict}
\title{A support vector machine style binary classifier}

\description{
	This function classifies points in a data set with a support vector machine
	using a GPU.  The negative category is represented by -1.f and the
	positive one by 1.f.  The underlying code is adapted from Austin
	Carpenter's cuSVM which can be found at
	\url{http://patternsonascreen.net/cuSVM.html}
}

\usage{
gpuSvmPredict(data, supportVectors, svCoefficients, svOffset,
	kernelWidth = 0.125, isRegression = FALSE)
}

\arguments{
	\item{data}{a matrix of floating point numbers.  Each row will be placed
		into one of two categories -1.f or 1.f.  Note that ncol(data) should
		equal ncol(supportVectors).}
	\item{supportVectors}{a matrix of floating point numbers.  Each row of the
		matrix is a support vector.  This matrix can be obtained from
		gpuSvmTrain, for example.  Note that ncol(supportVector) should equal
		ncol(data).}
	\item{svCoefficients}{a vector of floating point numbers representing 
		coefficients corresponding to the support vectors.  This vector can be
		obtained from gpuSvmTrain, for example.  Each support vector
		supportVectors[i,] has coefficient svCoefficients[i].}
	\item{svOffset}{a single floating point number.  It is the offset for the
		prediction function.  The offset can be obtained from gpuSvmTrain, for
		example.}
	\item{kernelWidth}{a single floating point number.  This is the scalar 
		Gaussian kernel parameter.}
	\item{isRegression}{a single logical value indicating if the supportVectors
		result from regression.}
}

\value{
a vector of nrow(data) entries, each either -1.f or 1.f.  Each entry i
corresponds to the support vector machine's prediction for the category of
data[i,].
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
# 
# b <- gpuSvmPredict(x, a$supportVectors, a$svCoefficients, a$svOffset, isRegression = FALSE)
# print(b)

# this time around, y : -1 or 1 and we set isRegression to FALSE
y <- runif(100, min = -1, max = 1)

x <- matrix(runif(500), 100, 5)

# a <- gpuSvmTrain(y, x, isRegression = TRUE)
# print(a)
# 
# b <- gpuSvmPredict(x, a$supportVectors, a$svCoefficients, a$svOffset, isRegression = TRUE)
# print(b)
}
