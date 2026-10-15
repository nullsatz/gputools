\name{gpuKmeans}
\alias{gpuKmeans}
\title{K-Means Clustering With a GPU}

\description{
	Partitions the rows of \code{x} into \code{k} clusters by Lloyd's
	algorithm, as \code{\link{kmeans}(x, centers, algorithm = "Lloyd")}
	does, with the distances and the cluster means computed on the GPU. It
	is for many points and many clusters: hundreds of thousands of cells
	of a single-cell experiment into hundreds of clusters, say.
}

\usage{
gpuKmeans(x, centers, iter.max = 10L, nstart = 1L,
		precision = getOption("gputools.precision", "single"))
}

\arguments{
	\item{x}{a numeric matrix, a row per point, with no missing values.}
	\item{centers}{the number of clusters, \code{k}, or a \code{k}-row
		matrix of distinct starting centres.}
	\item{iter.max}{the most iterations to run.}
	\item{nstart}{with a number of clusters, how many random starts to try;
		the best (least total within-cluster sum of squares) is kept.}
	\item{precision}{\code{"single"} or \code{"double"}: the floating point
		precision the GPU computes in; see \code{\link{gpuCor}}.}
}

\details{
	Each iteration assigns every point to its nearest centre and moves each
	centre to the mean of its points, until no assignment changes. The
	random starts are chosen exactly as \code{kmeans} chooses them, so with
	the same seed the two start from the same centres, and with the same
	starts they take the same steps: the results agree, apart from a point
	equally near two centres, which rounding can send either way (more
	often in single precision).

	\code{kmeans}' default is the Hartigan-Wong algorithm, which also moves
	single points between clusters and usually ends at a lower within-cluster
	sum of squares from the same start; Lloyd's is the one that parallelises.
	A few random starts (\code{nstart}) make up much of the difference.

	As \code{kmeans} does, it warns of a cluster left empty (its centre is
	then \code{NaN}) and of not converging in \code{iter.max} iterations.
}

\value{
	An object of class \code{"kmeans"}, as \code{\link{kmeans}} returns,
	so its print and \code{fitted} methods apply.
}

\seealso{
	\code{\link{kmeans}}
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
set.seed(1)
x <- rbind(matrix(rnorm(2000, sd = 0.3), ncol = 2),
           matrix(rnorm(2000, mean = 2, sd = 0.3), ncol = 2))
cl <- gpuKmeans(x, 2, nstart = 5)
cl$size
cl$centers

# the same starts give the same clustering as kmeans()
set.seed(2)
g <- gpuKmeans(x, 4, precision = "double")
set.seed(2)
k <- kmeans(x, 4, algorithm = "Lloyd")
all(g$cluster == k$cluster)
\dontshow{\}) # examplesIf}
}
