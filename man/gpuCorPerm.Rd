\name{gpuCorPerm}
\alias{gpuCorPerm}
\title{Permutation Tests of Correlations, With a GPU}

\description{
	Tests every correlation between the columns of \code{x} and those of
	\code{y} (or among the columns of \code{x}) by permutation: the rows of
	\code{y} are shuffled \code{B} times and each correlation's p-value is
	how often a shuffle gives one as strong. It also gives p-values
	adjusted for testing all the pairs at once, by Westfall and Young's
	maxT. Permutation p-values need no assumption of normality, which for
	small samples and skewed data the t test of \code{\link{gpuCor}}
	makes; each shuffle costs a whole correlation matrix, which the GPU
	computes in one matrix product.
}

\usage{
gpuCorPerm(x, y = NULL, B = 1000L, method = "pearson",
		precision = getOption("gputools.precision", "single"))
}

\arguments{
	\item{x}{a numeric matrix whose columns are the variables, with no
		missing values.}
	\item{y}{a numeric matrix with the same number of rows, or \code{NULL}
		for \code{x} with itself.}
	\item{B}{the number of permutations. The smallest p-value possible is
		\eqn{1 / (B + 1)}.}
	\item{method}{\code{"pearson"} or \code{"spearman"}.}
	\item{precision}{\code{"single"} or \code{"double"}: the floating point
		precision the GPU computes in; see \code{\link{gpuCor}}.}
}

\details{
	The permutations are drawn with \code{\link{sample.int}}, so
	\code{set.seed} makes the results reproducible. Each is applied to the
	rows of \code{y} (of a copy of \code{x} when \code{y} is \code{NULL}),
	and all of a permutation's correlations are computed from it, so the
	tests share their permutations, as the maxT adjustment needs.

	The p-value of a correlation \eqn{r} is \eqn{(1 + b) / (1 + B)}, where
	\eqn{b} of the \eqn{B} permutations gave a correlation with
	\eqn{|r_b| \ge |r|}{|r_b| >= |r|}: two-sided, and counting the observed
	data as one of the permutations (Phipson and Smyth 2010), so that it is
	never zero. The adjusted p-value counts instead the permutations whose
	largest \eqn{|r_b|} over all the tests reaches \eqn{|r|}: single-step
	maxT (Westfall and Young 1993), which controls the family-wise error
	rate. A permuted \eqn{|r_b|} within a relative allowance of \eqn{|r|}
	(1e-10 in double, 1e-5 in single) counts as reaching it, so that a
	permutation equivalent to the observed data always counts.

	The whole \code{ncol(x)} by \code{ncol(y)} matrix is held on the GPU,
	twice; for wider data, split the columns of \code{y} (the maxT
	adjustment is then within each part).
}

\value{
	A list:
	\item{coefficients}{the correlations, as \code{cor} gives them.}
	\item{p.value}{the permutation p-values.}
	\item{p.adjusted}{the maxT adjusted p-values.}
	\item{permutations}{\code{B}.}
	With \code{y = NULL} the diagonal is \code{NA}.
}

\references{
	Westfall, P. H. and Young, S. S. (1993) \emph{Resampling-Based Multiple
	Testing}. Wiley.

	Phipson, B. and Smyth, G. K. (2010) Permutation p-values should never
	be zero. \emph{Statistical Applications in Genetics and Molecular
	Biology} \bold{9}, 39.
}

\seealso{
	\code{\link{gpuCor}}, \code{\link{cor.test}}
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
set.seed(1)
# 15 samples of skewed data; one real association
x <- matrix(rexp(15 * 40), 15, 40)
y <- matrix(rexp(15 * 30), 15, 30)
y[, 1] <- y[, 1] + 2 * x[, 1]
res <- gpuCorPerm(x, y, B = 2000)
res$p.value[1, 1]
res$p.adjusted[1, 1]
min(res$p.adjusted[-1, -1])
\dontshow{\}) # examplesIf}
}
