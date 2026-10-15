\name{gpuHclust}
\alias{gpuHclust}
\title{Perform Hierarchical Clustering for Vectors with a GPU}

\description{
	This function performs clustering on a set of points.  The distance between
	each pair of points should be calculated first using a function like 
	'gpuDist' or 'dist'.
}

\usage{
	gpuHclust(distances, method = "complete", beta = 0.5,
		precision = getOption("gputools.precision", "single"))
}

\arguments{
	\item{distances}{a class of type "dist" containing floating point numbers
		representing distances between points.  R's native dist function and
		the gpuDist function produce output of this type.}  
	\item{method}{a string representing the name of the clustering method to be
		applied to distances.  Currently supported method names include
		"average", "centroid", "complete", "flexible", "flexible group", 
		"mcquitty", "median", "single", "ward", and "wpgma".}
	\item{beta}{the parameter of the \code{"flexible"} and
		\code{"flexible group"} methods, ignored by the others; it must be
		below 1. The flexible-beta update of Lance and Williams (1967) is
		\deqn{d(k, i \cup j) = \frac{1 - \beta}{2}\left(d(k,i) + d(k,j)\right) + \beta\, d(i,j);}{d(k, i+j) = (1 - beta)/2 (d(k,i) + d(k,j)) + beta d(i,j);}
		flexible UPGMA (\code{"flexible group"}, Belbin, Faith and Milligan
		1992) weights \eqn{d(k,i)} and \eqn{d(k,j)} by the sizes of clusters
		\eqn{i} and \eqn{j} instead of equally. The default, 0.5, is the value
		earlier versions of gputools used (it was fixed then).
		Lance and Williams recommend a small negative value,
		\code{beta = -0.25}, which is the usual choice in the literature;
		positive values contract the space, so clusters tend to chain.}
	\item{precision}{\code{"single"} or \code{"double"}: the floating point precision
		the GPU computes in. Single precision is the default, set with
		\code{options(gputools.precision = "double")}; it runs many times
		faster on most consumer GPUs. Results are returned as doubles either way.}
}

\value{Copied from the native R function 'hclust' documentation.  A class of
	type "hclust" with the following attributes. 
	\item{merge}{an n-1 by 2 matrix. Row i of 'merge' describes the merging of
		clusters at step i of the clustering. If an element j in the
		row is negative, then observation -j was merged at this
		stage. If j is positive then the merge was with the cluster
		formed at the (earlier) stage j of the algorithm. Thus
		negative entries in 'merge' indicate agglomerations of
		singletons, and positive entries indicate agglomerations of
		non-singletons. Copied from the native R function 'hclust'
		documentation.}
	\item{order}{a vector giving the permutation of the original observations
		suitable for plotting, in the sense that a cluster plot using
		this ordering and matrix 'merge' will not have crossings of
		the branches.}
	\item{height}{a set of n-1 non-decreasing real values. The clustering
		height: that is, the value of the criterion associated with
		the clustering 'method' for the particular agglomeration.}
}

\references{
	Lance, G. N. and Williams, W. T. (1967). A general theory of
	classificatory sorting strategies: 1. Hierarchical systems.
	\emph{The Computer Journal}, 9(4), 373--380.

	Belbin, L., Faith, D. P. and Milligan, G. W. (1992). A comparison of two
	approaches to beta-flexible clustering. \emph{Multivariate Behavioral
	Research}, 27(3), 417--433.
}
\seealso{
	hclust, \code{\link{gpuDistClust}}
}

\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
numVectors <- 5
dimension <- 10
Vectors <- matrix(runif(numVectors*dimension), numVectors, dimension)
distMat <- gpuDist(Vectors, "euclidean")
myClust <- gpuHclust(distMat, "single")
plot(myClust)
\dontshow{\}) # examplesIf}
}

\keyword{cluster}
