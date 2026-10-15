gpuGranger <- function(x, y=NULL, lag,
    precision = getOption("gputools.precision", "single"))
{
    precision <- .precision(precision)
    x <- as.matrix(x)
    rows <- nrow(x)
    colsx <- ncol(x)

    if(rows - lag <= 2 * lag + 1) {
        stop("time sequence too short for lag: use longer sequences or smaller lag")
    }

    if(!is.null(y)) {
        y <- as.matrix(y)
    }

    lag <- as.integer(lag)

    if(is.null(y)) {
        colsy <- colsx
        cRetVal <- .Call(C_gpuGranger, x, as.integer(rows), as.integer(colsx),
                         lag, precision)
    } else {
        colsy <- ncol(y)
        cRetVal <- .Call(C_gpuGrangerXY, x, as.integer(rows),
                         as.integer(colsx), y, as.integer(colsy), lag,
                         precision)
    }
    # [i, j] tests whether x[, i] Granger-causes x[, j] (or y[, j]); the
    # dimnames say so when printed
    names <- function(m, prefix)
        if (is.null(colnames(m))) paste0(prefix, seq_len(ncol(m))) else colnames(m)
    causes <- names(x, "x")
    effects <- if (is.null(y)) causes else names(y, "y")
    dn <- list(cause = causes, effect = effects)
    fStats <- matrix(cRetVal$fStats, colsx, colsy, dimnames = dn)
    pValues <- matrix(cRetVal$pValues, colsx, colsy, dimnames = dn)
    if (is.null(y)) {
        # a series does not Granger-cause itself: no test, so no value
        diag(fStats) <- NA
        diag(pValues) <- NA
    }
    return(list(fStatistics = fStats, pValues = pValues))
}
