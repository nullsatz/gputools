gpuGranger <- function(x, y=NULL, lag)
{
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
                         lag)
    } else {
        colsy <- ncol(y)
        cRetVal <- .Call(C_gpuGrangerXY, x, as.integer(rows),
                         as.integer(colsx), y, as.integer(colsy), lag)
    }
    fStats <- matrix(cRetVal$fStats, colsx, colsy)
    pValues <- matrix(cRetVal$pValues, colsx, colsy)
    return(list(fStatistics = fStats, pValues = pValues))
}
