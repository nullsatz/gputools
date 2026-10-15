gpuAssoc <- function(x, y, covariates = NULL, pvalue = 1e-5, top = NULL,
    limit = 1e7, tile = 2048L,
    precision = getOption("gputools.precision", "single")) {
    precision <- .precision(precision)
    x <- as.matrix(x)
    y <- as.matrix(y)
    n <- nrow(x)
    if(nrow(y) != n) {
        stop("x and y have different numbers of rows")
    }
    if(!is.null(covariates)) {
        covariates <- as.matrix(covariates)
        if(nrow(covariates) != n) {
            stop("covariates has a different number of rows from x")
        }
    }
    if(anyNA(x) || anyNA(y) || anyNA(covariates)) {
        stop("x, y and covariates must have no missing values")
    }
    if(!is.null(top)) {
        if(!(is.numeric(top) && length(top) == 1 && !is.na(top) &&
             top >= 1)) {
            stop("top must be a positive whole number")
        }
    } else if(!(is.numeric(pvalue) && length(pvalue) == 1 &&
                !is.na(pvalue) && pvalue > 0 && pvalue <= 1)) {
        stop("pvalue must be a number in (0, 1]")
    }
    tile <- as.integer(tile)
    if(length(tile) != 1 || is.na(tile) || tile < 1) {
        stop("tile must be a positive whole number")
    }

    # an orthonormal basis for the intercept and covariates: the part of x
    # and y in it is projected out of both on the GPU
    design <- cbind(rep(1, n), covariates)
    storage.mode(design) <- "double"
    qrd <- qr(design)
    if(qrd$rank < ncol(design)) {
        stop("the covariates are collinear (with each other or the intercept)")
    }
    q <- qr.Q(qrd)
    df <- n - 1L - ncol(design)
    if(df < 1) {
        stop("too few rows: need more than ", ncol(design) + 1,
             " for an intercept, ", ncol(design) - 1, " covariate(s) and x")
    }

    xNames <- colnames(x)
    yNames <- colnames(y)
    storage.mode(x) <- "double"
    storage.mode(y) <- "double"

    # p <= pvalue is |r| >= rc; the GPU's correlations are rounded, so it
    # takes a little more and the exact test is applied here
    if(is.null(top)) {
        tc <- qt(pvalue / 2, df, lower.tail = FALSE)
        rc <- tc / sqrt(df + tc^2)
        slack <- if(precision == "double") 1e-9 else 1e-4
        mode <- 0L
        threshold <- rc * (1 - slack)
    } else {
        mode <- 1L
        threshold <- 0
    }
    ans <- .Call(C_gpuAssoc, x, y, q, mode, as.double(threshold),
                 as.integer(if(is.null(top)) 0 else top), tile,
                 as.double(limit), precision)

    r <- ans$r
    t <- .tScore(r, df + 2)
    p <- 2 * pt(-abs(t), df)
    res <- data.frame(i = ans$i, j = ans$j)
    if(!is.null(xNames)) res$x <- xNames[res$i]
    if(!is.null(yNames)) res$y <- yNames[res$j]
    res$beta <- r * ans$ylen[res$j] / ans$xlen[res$i]
    res$t <- t
    res$p.value <- p
    if(mode == 0L) {
        keep <- p <= pvalue
        res <- res[keep, , drop = FALSE]
        res <- res[order(res$p.value, res$j, res$i), , drop = FALSE]
    }
    # Benjamini-Hochberg over all the tests made, from the reported ones
    # only: it can overstate the FDR, never understate it
    ntests <- as.double(ncol(x)) * ncol(y)
    o <- order(res$p.value)
    fdr <- numeric(nrow(res))
    fdr[o] <- pmin(1, rev(cummin(rev(res$p.value[o] * ntests /
                                     seq_along(o)))))
    res$fdr <- fdr
    rownames(res) <- NULL
    res
}
