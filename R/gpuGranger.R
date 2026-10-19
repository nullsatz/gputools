gpuGranger <- function(x, y=NULL, lag,
    precision = getOption("gputools.precision", "single"))
{
    # the argument is accepted for the common interface; the Granger tests
    # always compute in double (see the help page)
    precision <- .precision(precision)
    x <- as.matrix(x)
    rows <- nrow(x)
    colsx <- ncol(x)

    if (ncol(x) < 1) stop("x needs at least one column")
    if (length(lag) != 1 || !is.numeric(lag) || is.na(lag) ||
        lag != round(lag) || lag < 1) {
        stop("lag must be a positive whole number")
    }
    if (lag > 16) {
        stop("lag must be at most 16")
    }
    lag <- as.integer(lag)
    if (rows - lag <= 2 * lag + 1) {
        stop("time sequence too short for lag: use longer sequences or smaller lag")
    }

    if(!is.null(y)) {
        y <- as.matrix(y)
        if (nrow(y) != rows) stop("x and y have different numbers of rows")
    }

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
    if (cRetVal$unresolved > 0) {
        warning(sprintf(paste(
            "%d pair(s) are too ill-conditioned to resolve in double",
            "precision: a cause almost exactly a linear function of the",
            "effect's own past, or nearly collinear lags (a series integrated",
            "twice or more: difference it; or an exactly periodic one); their",
            "tests are NA"), cRetVal$unresolved), call. = FALSE)
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
    # no test (a series against itself, a constant series) or no answer
    # (unresolvable) is NA, not NaN, as R reports a missing statistic
    fStats[is.nan(fStats)] <- NA
    pValues[is.nan(pValues)] <- NA
    return(list(fStatistics = fStats, pValues = pValues))
}
