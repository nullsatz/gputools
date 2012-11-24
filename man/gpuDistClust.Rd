\name{gpuDistClust}
\alias{gpuDistClust}
\title{Compute Distances and Hierarchical Clustering for Vectors on a GPU}

\description{This function takes a set of vectors and performs clustering
	on them.  The function will first calculate the distance between all of the 
	pairs of vectors and then use the distances to cluster the vectors.  Both of
	these steps are done on the GPU.
}

\usage{
	gpuDistClust(points, distmethod = "euclidean", clustmethod = "complete") 
}

\arguments{
	\item{points}{a matrix of floating point numbers in which each row is a
		vector in $R^n$ space where $n$ is ncol(points).}
	\item{distmethod}{a string representing the name of the metric to use to
		calculate the distance between the vectors of 'points'.  Currently
		supported values are: "binary", "canberra", "euclidean", "manhattan",
		"maximum".}
	\item{clustmethod}{a string representing the name of the clustering method 
		to be applied to distances.  Currently supported method names include
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
	\code{\link{gpuDist}}, \code{\link{gpuHclust}}.
}

\examples{
numVectors <- 5
dimension <- 10
Vectors <- matrix(runif(numVectors*dimension), numVectors, dimension)
myClust <- gpuDistClust(Vectors, "maximum", "mcquitty")
plot(myClust)
}

\keyword{cluster}
