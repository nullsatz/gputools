\name{gpuGranger}
\alias{gpuGranger}
\title{Perform Granger Causality Tests for Vectors on a GPU}

\description{
	This function performs, with the aid of a GPU, Granger Causality Tests on 
	permutations of pairs of columns of the input matrices 'x' and 'y'.
}

\usage{
gpuGranger(x, y=NULL, lag)
}

\arguments{
	\item{x}{a matrix of floating point values.  Each column represents
		a sequence of observations for a single random variable.}
	\item{y}{an optional  matrix of floating point values.  Each column
		represents a sequence of observations for a single random variable.}
	\item{lag}{a positive integer by which to offset the sequence of 
		observations to calculate the coefficient for Granger causality.}
}

\value{a list of two single precision floating point matrices both of the same
	dimension. The two matrices are fStatistics and pValues. The
	fStatistics matrix holds the F-statistics from the Granger causality tests.
	Each element of the pValues matrix is the p-value for the corresponding
	element of the fStatistics matrix.

	If y is NULL, the test is run on permutations of pairs of columns of x. To
	find the Granger causality F-statistic estimating the answer to
	"Does variable x[ ,j] Granger-cause variable x[ ,i]?", look at
	fStatistics[i, j] and pValues[i, j].

	If y is not NULL, the test is run on permutations of pairs (x[ ,i], y[ ,j]).
	To find the Granger causality F-statistic estimating the answer to
	"Does variable y[ ,j] Granger-cause variable x[ ,i]?", look at
	fStatistics[i, j] and pValues[i, j].
}

\examples{
# permutations of pairs of cols of just x
numRandVars <- 5
numSamples <- 20
randVarSequences <- matrix(runif(numRandVars*numSamples), numSamples,
	numRandVars)
gpuGranger(randVarSequences, lag = 5)

# pairs of cols, one from x and one from y
numXRandVars <- 5
numXSamples <- 20
x <- matrix(runif(numXRandVars*numXSamples), numXSamples, numXRandVars)

numYRandVars <- 3
numYSamples <- 20
y <- matrix(runif(numYRandVars*numYSamples), numYSamples, numYRandVars)

result <- gpuGranger(x, y, lag = 5)
print(result)
}
