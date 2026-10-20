\name{gpuRowTtest}
\alias{gpuRowTtest}
\title{Two-Group t Tests of Every Row, With Permutation p-Values on a GPU}

\description{
	Tests every row of \code{x} (every gene of an expression matrix, say)
	for a difference in mean between two groups of columns, as
	\code{\link{t.test}} does one row at a time, and optionally gives each
	test a permutation p-value and a p-value adjusted for testing all the
	rows, by Westfall and Young's maxT. The permutations run on the GPU,
	every row under every relabelling of the samples at once.
}

\usage{
gpuRowTtest(x, group, var.equal = FALSE, B = 0L,
		precision = getOption("gputools.precision", "single"))
}

\arguments{
	\item{x}{a numeric matrix, a row per variable (gene), a column per
		sample, with no missing values.}
	\item{group}{the samples' groups: a vector or factor with one entry per
		column of \code{x} and exactly two levels, at least two samples in
		each.}
	\item{var.equal}{\code{FALSE} for Welch's test, \code{TRUE} for
		Student's with the variances pooled, as in \code{t.test}.}
	\item{B}{the number of permutations, or 0 for none.}
	\item{precision}{\code{"single"} or \code{"double"}: the floating point
		precision the GPU computes the permutations in; see
		\code{\link{gpuCor}}.}
}

\details{
	The t statistics, degrees of freedom and parametric p-values are those
	of \code{t.test(x[i, group == level1], x[i, group == level2],
	var.equal = var.equal)}, the first level's mean minus the second's.
	They are computed in double precision on the CPU: a t test of each row
	is too little work to be worth sending to the GPU.

	With \code{B > 0}, the samples' labels are permuted \code{B} times
	(drawn with \code{\link{sample.int}}, so \code{set.seed} makes them
	reproducible) and every row's t statistic is computed under each
	permutation, on the GPU: two matrix products give every row's group
	sums and sums of squares under all the labellings. A row's permutation
	p-value is \eqn{(1 + b) / (1 + B)} with \eqn{b} the permutations giving
	\eqn{|t_b| \ge |t|}{|t_b| >= |t|} (Phipson and Smyth 2010); its
	adjusted p-value counts the permutations whose largest \eqn{|t_b|} over
	all the rows reaches \eqn{|t|}: single-step maxT, which controls the
	family-wise error rate (Westfall and Young 1993). Permuting the labels
	keeps the dependence between genes, which the adjustment exploits, and
	makes no assumption of normality.

	A permuted \eqn{|t_b|} within a relative allowance of \eqn{|t|}
	(1e-10 in double, 1e-5 in single) counts as reaching it, so that a
	relabelling equivalent to the observed one always counts; with equal
	groups a labelling and its swap are made the same labelling, as they
	are the same test. A row constant within each group but with the
	groups' means apart separates them completely: its statistic is
	\code{Inf} or \code{-Inf} and its p-values 0 (as SciPy reports it),
	and in the permutations every relabelling that separates the groups
	as completely counts. A constant row, or one constant within both
	groups with equal means, has no test: its statistic is \code{NaN} and
	its p-values \code{NA}.
}

\value{
	A data frame, a row per row of \code{x}, named as \code{x}'s rows:
	\item{mean1, mean2}{the groups' means.}
	\item{statistic}{the t statistic.}
	\item{df}{its degrees of freedom.}
	\item{p.value}{the two-sided p-value from the t distribution.}
	\item{p.perm}{with \code{B > 0}, the permutation p-value.}
	\item{p.adjusted}{with \code{B > 0}, the maxT adjusted p-value.}
}

\references{
	Westfall, P. H. and Young, S. S. (1993) \emph{Resampling-Based Multiple
	Testing}. Wiley.

	Phipson, B. and Smyth, G. K. (2010) Permutation p-values should never
	be zero. \emph{Statistical Applications in Genetics and Molecular
	Biology} \bold{9}, 39.
}

\seealso{
	\code{\link{t.test}}, \code{\link{gpuCorPerm}}
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
set.seed(1)
genes <- matrix(rnorm(2000 * 12), 2000, 12,
                dimnames = list(paste0("g", 1:2000), NULL))
group <- rep(c("treated", "control"), each = 6)
genes[1:5, group == "treated"] <- genes[1:5, group == "treated"] + 3
res <- gpuRowTtest(genes, group, B = 5000)
head(res[order(res$p.adjusted), ])
\dontshow{\}) # examplesIf}
}
