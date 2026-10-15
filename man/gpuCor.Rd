\name{gpuCor}
\alias{gpuCor}
\title{Calculate Various Correlation Coefficients With a GPU}

\description{
	Correlates each column of \code{x} with each column of \code{y}, as
	\code{\link{cor}} does, with Pearson's, Kendall's (tau-b) or
	Spearman's coefficient, and tests each correlation for zero, as
	\code{\link{cor.test}} does.
}

\usage{
gpuCor(x, y = NULL, use = "everything", method = "pearson",
		precision = getOption("gputools.precision", "single"))
}

\arguments{
	\item{x}{a numeric matrix whose columns are the variables.}
	\item{y}{a numeric matrix with the same number of rows, or \code{NULL}
		for \code{x} with itself.}
	\item{use}{\code{"everything"} (a variable with a missing value gives
		\code{NA}) or \code{"pairwise.complete.obs"} (each pair of variables
		uses the rows where both are present), as with \code{cor}.}
	\item{method}{\code{"pearson"}, \code{"kendall"} or \code{"spearman"}.}
	\item{precision}{\code{"single"} or \code{"double"}: the floating point precision
		the GPU computes in. Single precision is the default, set with
		\code{options(gputools.precision = "double")}; it runs many times
		faster on most consumer GPUs. Results are returned as doubles either way.}
}

\details{
	Kendall's coefficient is tau-b and Spearman's is Pearson's on the ranks,
	ties averaged; with \code{use = "pairwise.complete.obs"}, each pair is
	ranked over its complete rows only. All three match \code{cor}.

	The p-values are two-sided and match \code{cor.test}: exactly for
	Pearson's (a t test on \code{pairs - 2} degrees of freedom), and as
	\code{cor.test(exact = FALSE)} gives them for Spearman's (the same t
	approximation) and Kendall's (a normal approximation with the tie
	correction). \code{cor.test}'s exact small-sample tests for the rank
	coefficients are not computed.
}

\value{
	A list of matrices, each with a row per column of \code{x} and a column
	per column of \code{y}, named as \code{cor} names them:
	\item{coefficients}{the correlations.}
	\item{ts}{for Pearson's and Spearman's, the t statistics,
		\eqn{r \sqrt{(n - 2) / (1 - r^2)}}{r sqrt((n - 2) / (1 - r^2))}.}
	\item{z}{for Kendall's, the normal statistics.}
	\item{p.value}{the two-sided p-values.}
	\item{pairs}{the number of rows each correlation used.}
}

\seealso{
	\code{\link{cor}}, \code{\link{cor.test}}
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
numAvars <- 5
numBvars <- 10
numSamples <- 30
A <- matrix(runif(numAvars*numSamples), numSamples, numAvars)
B <- matrix(runif(numBvars*numSamples), numSamples, numBvars)
gpuCor(A, B, method="pearson")
gpuCor(A, B, method="kendall")
gpuCor(A, B, method="spearman")$p.value
A[3,2] <- NA
gpuCor(A, B, use="pairwise.complete.obs", method="pearson")
\dontshow{\}) # examplesIf}
}
