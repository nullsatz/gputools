gpuMi <- function(x, y = NULL, bins = 10, splineOrder = 3,
    precision = getOption("gputools.precision", "single"))
{
    precision <- .precision(precision)
    x <- as.matrix(x)
    storage.mode(x) <- "double"
    if (!is.null(y)) {
        y <- as.matrix(y)
        storage.mode(y) <- "double"
        if (nrow(y) != nrow(x)) stop("x and y have different numbers of rows")
    }
    whole <- function(v) length(v) == 1 && is.numeric(v) && !is.na(v) &&
        v == round(v)
    if (!whole(bins) || !whole(splineOrder) || splineOrder < 1 ||
        splineOrder > bins) {
        stop("bins and splineOrder must be whole numbers with ",
             "1 <= splineOrder <= bins")
    }
    if (bins > 64) stop("bins must be at most 64")
    if (nrow(x) < 2) stop("need at least 2 rows")
    if (anyNA(x) || (!is.null(y) && anyNA(y))) {
        stop("x and y must have no missing values")
    }
    bins <- as.integer(bins)
    splineOrder <- as.integer(splineOrder)
    nsamples <- as.integer(nrow(x))
    na <- as.integer(ncol(x))
    nb <- if (is.null(y)) na else as.integer(ncol(y))

    mi <- .Call(C_gpuMutualInfo, bins, splineOrder, nsamples, x, na, y, nb,
                precision)
    mi <- matrix(mi, nb, na)
    rownames(mi) <- if (is.null(y)) colnames(x) else colnames(y)
    colnames(mi) <- colnames(x)
    mi
}
