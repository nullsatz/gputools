\name{gpuCor}
\alias{gpuCor}
\title{Calculate Various Correlation Coefficients With a GPU}

\description{
	The correlation coefficient will be calculated
	for each pair $x_i$, $y_j$ where $x_i$ is a column of $x$ and $y_j$ is a
	column of $y$.  Currently, Pearson's and Kendall's correlation
	coefficient are implemented.  Pearson's may be calculated for
	data sets containing NAs in which case, the implementation behaves
	as R-native cor function with use="pairwise.complete".
}

\usage{
gpuCor(x, y = NULL, use = "everything", method = "pearson")
}

\arguments{
	\item{x}{a matrix of floating point values in which each column is a 
		random variable.}  
	\item{y}{a matrix of floating point values in which each column is a 
		random variable.}
	\item{use}{a string.  A character string giving a method for computing
		in the presence of missing values. Options are
		"everything" or "pairwise.complete.obs". This currently only 
		affects the "pearson" method.}
	\item{method}{a string.  Either "pearson" or "kendall".}
}

\value{
	For method "pearson", a list with matrices 'pairs',
	'coefficients', and 'ts'. The matrix entry $i$, $j$ for pairs represents
	the number of pairs of entries $x_i^k$, $y_j^k$ (the $k$-th entry from
	$x_i$ and $y_j$ respectively). These are the
	number of entries actually used to calculate the coefficients.
	Entry $i$, $j$ of the coefficients matrix is the correlation coefficient
	for $x_i$, $y_j$. Entry $i$, $j$ of the ts matrix is the t-score of the
	$i$, $j$ entry of the coefficient matrix. If use="pairwise.complete.obs"
	then only the pairs where both entries are not NA are used in the
	computations.

	For method "kendall", a list of matrices 'pairs' as above and
	'coefficients' as follows. The matrix 'coefficients' is a matrix of
	floating point numbers where entry $i$, $j$ is the correlation coefficient
	for $x_i$, $y_j$. Calculation of t-scores for the kendall coefficients is
	not yet implemented.
}

\seealso{
	cor
}

\examples{
numAvars <- 5
numBvars <- 10
numSamples <- 30
A <- matrix(runif(numAvars*numSamples), numSamples, numAvars)
B <- matrix(runif(numBvars*numSamples), numSamples, numBvars)
gpuCor(A, B, method="pearson")
gpuCor(A, B, method="kendall")
A[3,2] <- NA
gpuCor(A, B, use="pairwise.complete.obs", method="pearson")
}
