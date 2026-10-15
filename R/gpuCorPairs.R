gpuCorPairs <- function(x, y = NULL, threshold = NULL, top = NULL,
    method = "pearson", absolute = TRUE, test = FALSE, limit = 1e7,
    tile = 2048L, precision = getOption("gputools.precision", "single")) {
    precision <- .precision(precision)
    x <- as.matrix(x)
    if(!is.null(y)) {
        y <- as.matrix(y)
        if(nrow(y) != nrow(x)) {
            stop("x and y have different numbers of rows")
        }
    }
    n <- nrow(x)
    if(n < 3) {
        stop("need at least 3 rows")
    }
    if(is.null(threshold) == is.null(top)) {
        stop("give exactly one of threshold and top")
    }
    if(!is.null(threshold) && !(is.numeric(threshold) &&
        length(threshold) == 1 && !is.na(threshold))) {
        stop("threshold must be a number")
    }
    if(!is.null(top) && !(is.numeric(top) && length(top) == 1 &&
        !is.na(top) && top >= 1)) {
        stop("top must be a positive whole number")
    }
    tile <- as.integer(tile)
    if(length(tile) != 1 || is.na(tile) || tile < 1) {
        stop("tile must be a positive whole number")
    }

    methods <- c("pearson", "spearman")
    method <- pmatch(method, methods, -1)
    if(method == -1) {
        stop("method must be \"pearson\" or \"spearman\"")
    }
    method <- methods[method]

    if(anyNA(x) || (!is.null(y) && anyNA(y))) {
        stop("x and y must have no missing values; for pairwise complete ",
             "correlations use gpuCor(use = \"pairwise.complete.obs\")")
    }
    xNames <- colnames(x)
    yNames <- if(is.null(y)) xNames else colnames(y)
    storage.mode(x) <- "double"
    if(!is.null(y)) storage.mode(y) <- "double"

    if(method == "spearman") {
        x <- .Call(C_gpuRank, x, precision)
        if(!is.null(y)) y <- .Call(C_gpuRank, y, precision)
    }

    mode <- if(is.null(threshold)) 1L else 0L
    ans <- .Call(C_gpuCorPairs, x, y, mode,
                 as.double(if(is.null(threshold)) 0 else threshold),
                 as.integer(if(is.null(top)) 0 else top),
                 isTRUE(absolute), tile, as.double(limit), precision)

    res <- data.frame(i = ans$i, j = ans$j, r = ans$r)
    # threshold pairs come back in the order the GPU found them
    if(mode == 0L) res <- res[order(res$j, res$i), , drop = FALSE]
    rownames(res) <- NULL

    if(!is.null(xNames)) res$x <- xNames[res$i]
    if(!is.null(yNames)) res$y <- yNames[res$j]

    if(test) {
        res$t <- .tScore(res$r, n)
        res$p.value <- 2 * pt(-abs(res$t), n - 2)
    }
    res
}
