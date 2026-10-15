\name{gpuAssoc}
\alias{gpuAssoc}
\title{Association Scan: Every Column of x Against Every Column of y, With a GPU}

\description{
	Tests every column of \code{x} against every column of \code{y} by
	linear regression, \code{lm(y[, j] ~ x[, i] + covariates)}, and returns
	the significant associations. It is the scan of expression quantitative
	trait loci (eQTL) studies, genotypes against gene expression, and of
	other many-by-many association screens; it computes what Matrix eQTL's
	linear model does (Shabalin 2012), on the GPU.
}

\usage{
gpuAssoc(x, y, covariates = NULL, pvalue = 1e-5, top = NULL,
		limit = 1e7, tile = 2048L,
		precision = getOption("gputools.precision", "single"))
}

\arguments{
	\item{x}{a numeric matrix, a column per tested variable (a SNP's
		genotypes, say, coded 0, 1, 2), a row per sample.}
	\item{y}{a numeric matrix, a column per outcome (a gene's expression),
		the same rows.}
	\item{covariates}{\code{NULL} or a numeric matrix of covariates to
		adjust for (sex, age, principal components of ancestry, ...), the
		same rows. An intercept is always included.}
	\item{pvalue}{report every association with a p-value at most this.}
	\item{top}{or instead, report each column of \code{y}'s \code{top}
		most significant columns of \code{x}; \code{pvalue} is then not
		used.}
	\item{limit}{the most associations to return: a \code{pvalue} that lets
		more through is an error, not a result of unexpected size.}
	\item{tile}{the block size the GPU works in; see
		\code{\link{gpuCorPairs}}. It has no effect on the result.}
	\item{precision}{\code{"single"} or \code{"double"}: the floating point
		precision the GPU computes in; see \code{\link{gpuCor}}.}
}

\details{
	For each pair, the slope of \code{x[, i]} in the least-squares fit of
	\code{y[, j]} on it, an intercept and the covariates is tested for zero
	with the t test \code{summary(lm(...))} reports, on \eqn{n - 2 - k}
	degrees of freedom for \eqn{n} samples and \eqn{k} covariates. As in
	Matrix eQTL, the intercept and covariates are projected out of every
	column of \code{x} and \code{y} once, after which each test is a
	correlation: the scan is one large matrix product, computed a block at a
	time and filtered on the GPU.

	A column of \code{x} or \code{y} that the covariates explain completely,
	or a constant one, has no test and is never reported. Missing values
	are not allowed: impute them first (Matrix eQTL uses the SNP's mean
	genotype) or drop the sample.

	The whole of \code{x} and \code{y} is copied to the GPU. For more SNPs
	than fit, call \code{gpuAssoc} on a chromosome or a block of SNPs at a
	time: the results are the same apart from the false discovery rates,
	which then count only that call's tests.
}

\value{
	A data frame, a row per reported association:
	\item{i}{the column of \code{x}.}
	\item{j}{the column of \code{y}.}
	\item{x, y}{the columns' names, if they have them.}
	\item{beta}{the slope of \code{x[, i]}.}
	\item{t}{its t statistic.}
	\item{p.value}{its two-sided p-value.}
	\item{fdr}{the Benjamini-Hochberg false discovery rate over all
		\code{ncol(x) * ncol(y)} tests. It is computed from the reported
		p-values alone, as Matrix eQTL computes it, and so can overstate the
		rate but never understate it.}
	With \code{pvalue}, rows are ordered from the most significant; with
	\code{top}, by \code{j} and then from the most significant.
}

\references{
	Shabalin, A. A. (2012) Matrix eQTL: ultra fast eQTL analysis via large
	matrix operations. \emph{Bioinformatics} \bold{28}, 1353--1358.
}

\seealso{
	\code{\link{lm}}, \code{\link{gpuCorPairs}}
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
set.seed(1)
samples <- 200
snps <- matrix(rbinom(samples * 1000, 2, 0.3), samples, 1000,
               dimnames = list(NULL, paste0("rs", 1:1000)))
age <- rnorm(samples, 50, 10)
genes <- matrix(rnorm(samples * 300), samples, 300,
                dimnames = list(NULL, paste0("gene", 1:300)))
genes[, 7] <- genes[, 7] + 0.6 * snps[, 42] + 0.05 * age   # one real eQTL

gpuAssoc(snps, genes, covariates = age, pvalue = 1e-6)

# the same test by lm()
summary(lm(genes[, 7] ~ snps[, 42] + age))$coefficients[2, ]
\dontshow{\}) # examplesIf}
}
