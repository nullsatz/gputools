\name{gpuHclust}
\alias{gpuHclust}
\title{Perform Hierarchical Clustering for Vectors with a GPU}

\description{
	This function performs clustering on a set of points.  The distance between
	each pair of points should be calculated first using a function like 
	'gpuDist' or 'dist'.
}

\usage{
	gpuHclust(distances, method = "complete")
}

\arguments{
	\item{distances}{a class of type "dist" containing floating point numbers
		representing distances between points.  R's native dist function and
		the gpuDist function produce output of this type.}  
	\item{method}{a string representing the name of the clustering method to be
		applied to distances.  Currently supported method names include
		"average", "centroid", "complete", "flexible", "flexible group", 
		"mcquitty", "median", "single", "ward", and "wpgma".}
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

\seealso{
	hclust, \code{\link{gpuDistClust}}
}

\examples{
numVectors <- 5
dimension <- 10
Vectors <- matrix(runif(numVectors*dimension), numVectors, dimension)
distMat <- gpuDist(Vectors, "euclidean")
myClust <- gpuHclust(distMat, "single")
plot(myClust)
}

\keyword{cluster}
