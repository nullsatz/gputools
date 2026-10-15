\name{gpuCorPairs}
\alias{gpuCorPairs}
\title{The Strongest Correlations Among Many Variables, With a GPU}

\description{
	Finds the strongly correlated pairs among the columns of \code{x} (or
	between the columns of \code{x} and those of \code{y}) without forming
	the whole correlation matrix: every pair whose correlation reaches a
	threshold, or each variable's strongest partners. It is for data too
	wide for \code{\link{gpuCor}} or \code{\link{cor}}: tens of thousands of
	genes, whose correlation matrix would take gigabytes, reduced to the
	edge list of a co-expression network.
}

\usage{
gpuCorPairs(x, y = NULL, threshold = NULL, top = NULL,
		method = "pearson", absolute = TRUE, test = FALSE, limit = 1e7,
		tile = 2048L, precision = getOption("gputools.precision", "single"))
}

\arguments{
	\item{x}{a numeric matrix whose columns are the variables, with no
		missing values.}
	\item{y}{a numeric matrix with the same number of rows, or \code{NULL}
		for \code{x} with itself.}
	\item{threshold}{keep every pair whose correlation is at least this
		strong.}
	\item{top}{or instead, keep each column of \code{y}'s \code{top}
		strongest partners among the columns of \code{x}. Give one of
		\code{threshold} and \code{top}.}
	\item{method}{\code{"pearson"} or \code{"spearman"}.}
	\item{absolute}{\code{TRUE} to measure strength by \eqn{|r|}, so that
		strong negative correlations count; \code{FALSE} to keep positive
		ones only.}
	\item{test}{\code{TRUE} to add t statistics and two-sided p-values, as
		\code{\link{gpuCor}} computes them.}
	\item{limit}{the most pairs to return: a threshold that lets more
		through is an error, not a result of unexpected size. Raise it if
		you want them all.}
	\item{tile}{the GPU computes the correlation matrix in blocks of
		\code{tile} x \code{tile}. The default suits GPUs with a few
		gigabytes free; it has no effect on the result.}
	\item{precision}{\code{"single"} or \code{"double"}: the floating point
		precision the GPU computes in; see \code{\link{gpuCor}}.}
}

\details{
	The correlations are the ones \code{cor} computes (Spearman's is
	Pearson's on the ranks, ties averaged). A constant column correlates
	with nothing, as its correlations are \code{NA}, and is never
	returned.

	With \code{y = NULL}, a threshold returns each pair once, with
	\code{i < j}; \code{top} returns each column's partners, so a pair
	that is in both columns' lists appears twice, and a column is never
	its own partner.

	A p-value cut-off \eqn{\alpha} is a threshold on \eqn{|r|}, since every
	pair has the same \eqn{n}: \eqn{|r| \ge t / \sqrt{n - 2 + t^2}}{|r| >=
	t / sqrt(n - 2 + t^2)} with \eqn{t} the \eqn{1 - \alpha/2} quantile of
	the t distribution on \eqn{n - 2} degrees of freedom. Remember how many
	pairs are tested when choosing \eqn{\alpha}.

	The whole of \code{x} (and \code{y}) is copied to the GPU, in the
	precision computed in; the correlation matrix itself never is.
}

\value{
	A data frame, a row per pair:
	\item{i}{the column of \code{x}.}
	\item{j}{the column of \code{y} (of \code{x} when \code{y} is
		\code{NULL}).}
	\item{r}{their correlation.}
	\item{x, y}{the columns' names, if they have them.}
	\item{t, p.value}{with \code{test = TRUE}.}
	Rows are ordered by \code{j}, then by \code{i} for a threshold, and
	from strongest to weakest for \code{top}.
}

\seealso{
	\code{\link{gpuCor}} for the whole matrix.
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
set.seed(1)
genes <- matrix(rnorm(50 * 2000), 50, 2000,
                dimnames = list(NULL, paste0("g", 1:2000)))
genes[, 2] <- genes[, 1] + rnorm(50, sd = 0.1)

# every pair correlated at |r| >= 0.6
head(gpuCorPairs(genes, threshold = 0.6))

# each gene's three strongest partners, with p-values
head(gpuCorPairs(genes, top = 3, test = TRUE))
\dontshow{\}) # examplesIf}
}
