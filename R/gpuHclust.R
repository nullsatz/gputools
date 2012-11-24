gpuDist <- function(points, method = "euclidean", p = 2.0)
{
	if(!is.na(pmatch(method, "euclidian"))) {
		method <- "euclidean"
	}

    methods <- c("euclidean", "maximum", "manhattan", "canberra", "binary",
		"minkowski")
	method <- pmatch(method, methods) # hey presto method becomes an int
	if(is.na(method)) {
		stop("invalid distance method")
	}
    if(method == -1) {
		stop("ambiguous distance method")
	}
	method <- methods[method] # return method to a meaningful string

	points <- as.matrix(points)
	numPoints <- nrow(points)

	a <- .C("Rdistances",
		as.single(t(points)),
		as.integer(numPoints),
		as.integer(ncol(points)),
		d = single(numPoints * numPoints),
		method, as.single(p),
		PACKAGE='gputools')

	d <- as.dist(matrix(a$d, numPoints, numPoints))
	attr(d, "Labels") <- dimnames(points)[[1L]]
	attr(d, "method") <- method
	attr(d, "call") <- match.call()

	if(!is.na(pmatch(method, "minkowski"))) {
		attr(d, "p") <- p
	}

	return(d)
}

gpuHclust <- function(distances, method = "complete")
{
    methods <- c("ward", "single", "complete", "average", "mcquitty",
		"median", "centroid", "flexible", "flexible group", "wpgma")
    method <-  pmatch(method, methods) # method is now an integer
    if(is.na(method)) {
		stop("invalid clustering method")
	}
    if(method == -1) {
		stop("ambiguous clustering method")
	}
	method <- methods[method] # return method to a meaningful string

    n <- as.integer(attr(distances, "Size"))
    if(is.null(n)) {
		stop("invalid dissimilarities")
	}
    if(n < 2) {
        stop("must have n >= 2 objects to cluster")
	}

	len <- as.integer(n*(n-1)/2)
	if(length(distances) != len) {
		if (length(distances) < len) {
			stop("dissimilarities of improper length")
		} else {
			warning("dissimilarities of improper length")
		}
	}

	numpoints <- n
    a <- .C("Rhcluster",
		as.single(as.matrix(distances)),
		as.integer(numpoints),
		merge = integer(2*(numpoints-1)),
		order = integer(numpoints),
		val = single(numpoints-1),
		method,
		PACKAGE='gputools')

	merge <- matrix(a$merge, numpoints-1, 2)

	tree <- list(merge = merge, height= a$val, order = a$order,
		labels = attr(distances, "Labels"),
		method = method,
		call = match.call(),
		dist.method = attr(distances, "method"))

    class(tree) <- "hclust"
    return(tree)
}

gpuDistClust <- function(points, distmethod = "euclidean", 
	clustmethod = "complete") 
{
	if(!is.na(pmatch(distmethod, "euclidian"))) {
		method <- "euclidean"
	}

    methods <- c("euclidean", "maximum", "manhattan", "canberra", "binary",
		"minkowski")
	distmethod <- pmatch(distmethod, methods) # hey presto method becomes an int
	if(is.na(distmethod)) {
		stop("invalid distance method")
	}
    if(distmethod == -1) {
		stop("ambiguous distance method")
	}
	distmethod <- methods[distmethod] # return method to a meaningful string

    methods <- c("ward", "single", "complete", "average", "mcquitty",
		"median", "centroid", "flexible", "flexible group", "wpgma")
    clustmethod <-  pmatch(clustmethod, methods) # method is now an integer
    if(is.na(clustmethod)) {
		stop("invalid clustering method")
	}
    if(clustmethod == -1) {
		stop("ambiguous clustering method")
	}
	clustmethod <- methods[clustmethod] # return method to a meaningful string

	points <- as.matrix(points)
	nump <- nrow(points)

	a <- .C("Rdistclust",
		distmethod, clustmethod,
		as.single(t(points)),
		as.integer(nump),
		as.integer(ncol(points)),
		merge = integer(2*(nump-1)),
		order = integer(nump),
		val = single(nump-1),
		PACKAGE='gputools')

	merge <- matrix(a$merge, nump-1, 2)

    tree <- list(merge = merge, height = a$val, order = a$order,
		labels = dimnames(points)[[1L]],
 		method = clustmethod,
		call = match.call(),
		dist.method = distmethod)

    class(tree) <- "hclust"

    return(tree)
}
