gpuCor <- function(x, y = NULL, use = "everything", method = "pearson") {
    x <- as.matrix(x)
    nx <- ncol(x)
    size <- nrow(x)

    if(is.null(y)) {
        y <- x
    } else {
        y <- as.matrix(y)
    }
    ny <- ncol(y)

    n <- nx * ny

    methods <- c("pearson", "kendall")
    method <- pmatch(method, methods, -1)
    if(is.na(method)) {
        stop("invalid correlation method")
    }
    if(method == -1) {
        stop("ambiguous correlation method")
    }

    uses <- c("everything", "pairwise.complete.obs")
    use <- pmatch(use, uses, -1)
    if(is.na(use)) {
        stop("invalid correlation method")
    }
    if(use == -1) {
        stop("ambiguous correlation method")
    }

    if(methods[method] == "pearson") {
        answer <- .C("rpmcc", NAOK=TRUE,
                     as.integer(use - 1), as.single(x), as.integer(nx),
                     as.single(y), as.integer(ny), as.integer(size),
                     pairs = single(n), corr = single(n), ts = single(n),
                     PACKAGE='gputools')

        pairs <- t(matrix(answer$pairs, ny, nx))
        corr <- t(matrix(answer$corr, ny, nx))
        ts <- t(matrix(answer$ts, ny, nx))

        return(list(coefficients = corr, ts = ts, pairs = pairs))

    } else if(methods[method] == "kendall") {

        if(uses[use] != "everything") {
            warning("NA handling for Kendall's is not yet supported. Defaulting to using everything. Sorry for any inconvenience.")
        }

        a <- .C("RgpuKendall",
                as.single(x), nx, as.single(y), ny, 
                size, result = double(nx*ny),
                PACKAGE = "gputools")

        pairs <- matrix(size, nx, ny)
        return(list(coefficients = matrix(a$result, nx, ny), pairs = pairs))
    } else {
        stop("This correlation method is not yet supported.")
    }
}

gpuTtest <- function(goodPairs, coeffs) {
    goodPairs <- as.single(goodPairs)
    coeffs <- as.single(coeffs)

    n <- as.integer(length(goodPairs))

    .C("rtestT", NAOK = TRUE,
       goodPairs, coeffs, n,
       results = single(n),
       PACKAGE = 'gputools')$results
}
