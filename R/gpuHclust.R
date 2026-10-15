gpuDist <- function(points, method = "euclidean", p = 2.0,
	precision = getOption("gputools.precision", "single"))
{
    precision <- .precision(precision)
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

    a <- .Call(C_gpuDist, t(points), as.integer(numPoints),
               as.integer(ncol(points)), method, as.double(p), precision)

    d <- as.dist(matrix(a, numPoints, numPoints))
    attr(d, "Labels") <- dimnames(points)[[1L]]
    attr(d, "method") <- method
    attr(d, "call") <- match.call()

    if(!is.na(pmatch(method, "minkowski"))) {
        attr(d, "p") <- p
    }

    return(d)
}

gpuHclust <- function(distances, method = "complete", beta = 0.5,
	precision = getOption("gputools.precision", "single"))
{
    precision <- .precision(precision)
    beta <- .flexibleBeta(beta)
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
    a <- .Call(C_gpuHclust, as.matrix(distances), as.integer(numpoints),
               method, beta, precision)

    merge <- matrix(a$merge, numpoints-1, 2)

    tree <- list(merge = merge, height = a$height, order = a$order,
                 labels = attr(distances, "Labels"),
                 method = method,
                 call = match.call(),
                 dist.method = attr(distances, "method"))

    class(tree) <- "hclust"
    return(tree)
}

gpuDistClust <- function(points, distmethod = "euclidean", 
                         clustmethod = "complete", beta = 0.5,
                         precision = getOption("gputools.precision", "single"))
{
    precision <- .precision(precision)
    beta <- .flexibleBeta(beta)
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

    a <- .Call(C_gpuDistClust, distmethod, clustmethod, t(points),
               as.integer(nump), as.integer(ncol(points)), beta, precision)

    merge <- matrix(a$merge, nump-1, 2)

    tree <- list(merge = merge, height = a$height, order = a$order,
                 labels = dimnames(points)[[1L]],
                 method = clustmethod,
                 call = match.call(),
                 dist.method = distmethod)

    class(tree) <- "hclust"

    return(tree)
}

# The flexible methods' beta: their update is
# d(k, i+j) = (1 - beta)/2 (d(k,i) + d(k,j)) + beta d(i,j), which needs
# beta < 1.
.flexibleBeta <- function(beta)
{
    if (!is.numeric(beta) || length(beta) != 1L || !is.finite(beta) ||
        beta >= 1) {
        stop("beta must be a single number below 1")
    }
    as.double(beta)
}
