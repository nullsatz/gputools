gpuCor <- function(x, y = NULL, use = "everything", method = "pearson",
    precision = getOption("gputools.precision", "single")) {
    precision <- .precision(precision)
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
        answer <- .Call(C_gpuPmcc, as.integer(use - 1), x, as.integer(nx),
                        y, as.integer(ny), as.integer(size), precision)

        pairs <- t(matrix(answer$pairs, ny, nx))
        corr <- t(matrix(answer$corr, ny, nx))
        ts <- t(matrix(answer$ts, ny, nx))

        return(list(coefficients = corr, ts = ts, pairs = pairs))

    } else if(methods[method] == "kendall") {

        if(uses[use] != "everything") {
            warning("NA handling for Kendall's is not yet supported. Defaulting to using everything. Sorry for any inconvenience.")
        }

        result <- .Call(C_gpuKendall, x, as.integer(nx), y, as.integer(ny),
                        as.integer(size), precision)

        pairs <- matrix(size, nx, ny)
        return(list(coefficients = matrix(result, nx, ny), pairs = pairs))
    } else {
        stop("This correlation method is not yet supported.")
    }
}

gpuTtest <- function(goodPairs, coeffs,
    precision = getOption("gputools.precision", "single")) {
    precision <- .precision(precision)
    .Call(C_gpuTtest, as.double(goodPairs), as.double(coeffs), precision)
}
