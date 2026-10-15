gpuKmeans <- function(x, centers, iter.max = 10L, nstart = 1L,
    precision = getOption("gputools.precision", "single")) {
    precision <- .precision(precision)
    # the starting centres are chosen exactly as kmeans() chooses them, so
    # that the same seed gives the same starts
    x <- as.matrix(x)
    m <- as.integer(nrow(x))
    if(is.na(m)) stop("invalid nrow(x)")
    p <- as.integer(ncol(x))
    if(is.na(p)) stop("invalid ncol(x)")
    if(missing(centers)) stop("'centers' must be a number or a matrix")
    storage.mode(x) <- "double"
    if(anyNA(x)) stop("x must have no missing values")
    if(length(centers) == 1L) {
        k <- centers
        if(nstart == 1L) centers <- x[sample.int(m, k), , drop = FALSE]
        if(nstart >= 2L || any(duplicated(centers))) {
            cn <- unique(x)
            mm <- nrow(cn)
            if(mm < k) stop("more cluster centers than distinct data points.")
            centers <- cn[sample.int(mm, k), , drop = FALSE]
        }
    } else {
        centers <- as.matrix(centers)
        if(any(duplicated(centers))) stop("initial centers are not distinct")
        cn <- NULL
        k <- nrow(centers)
        if(m < k) stop("more cluster centers than data points")
    }
    k <- as.integer(k)
    if(is.na(k)) stop("invalid value of 'k'")
    iter.max <- as.integer(iter.max)
    if(is.na(iter.max) || iter.max < 1L) stop("'iter.max' must be positive")
    if(ncol(x) != ncol(centers)) {
        stop("must have same number of columns in 'x' and 'centers'")
    }
    storage.mode(centers) <- "double"

    # distances are found as |c|^2 - 2 x.c on the GPU; centring the data
    # first keeps that from cancelling away the precision of data far from
    # the origin, and moves no point nearer any centre
    mid <- colMeans(x)
    xc <- sweep(x, 2L, mid)

    doOne <- function(centers) {
        z <- .Call(C_gpuKmeans, xc, sweep(centers, 2L, mid), iter.max,
                   precision)
        z$centers <- sweep(z$centers, 2L, mid, "+")
        # each cluster's sum of squares about its final centre, in double
        d2 <- rowSums((x - z$centers[z$cluster, , drop = FALSE])^2)
        s <- rowsum(d2, z$cluster)
        z$wss <- numeric(k)
        z$wss[as.integer(rownames(s))] <- s[, 1]
        if(any(z$size == 0)) {
            warning("empty cluster: try a better set of initial centers",
                    call. = FALSE)
        }
        if(z$iter > iter.max) {
            warning(sprintf(ngettext(iter.max,
                "did not converge in %d iteration",
                "did not converge in %d iterations"), iter.max),
                call. = FALSE, domain = NA)
            z$ifault <- 2L
        }
        z
    }

    Z <- doOne(centers)
    best <- sum(Z$wss)
    if(nstart >= 2L && !is.null(cn)) {
        for(i in 2:nstart) {
            centers <- cn[sample.int(mm, k), , drop = FALSE]
            ZZ <- doOne(centers)
            if((z <- sum(ZZ$wss)) < best) {
                Z <- ZZ
                best <- z
            }
        }
    }
    centers <- matrix(Z$centers, k)
    dimnames(centers) <- list(1L:k, dimnames(x)[[2L]])
    cluster <- Z$cluster
    if(!is.null(rn <- rownames(x))) names(cluster) <- rn
    totss <- sum(xc^2)
    structure(list(cluster = cluster, centers = centers, totss = totss,
                   withinss = Z$wss, tot.withinss = best,
                   betweenss = totss - best, size = Z$size, iter = Z$iter,
                   ifault = Z$ifault),
              class = "kmeans")
}
