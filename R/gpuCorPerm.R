gpuCorPerm <- function(x, y = NULL, B = 1000L, method = "pearson",
    precision = getOption("gputools.precision", "single")) {
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
    B <- as.integer(B)
    if(length(B) != 1 || is.na(B) || B < 1) {
        stop("B must be a positive whole number")
    }
    methods <- c("pearson", "spearman")
    method <- pmatch(method, methods, -1)
    if(method == -1) {
        stop("method must be \"pearson\" or \"spearman\"")
    }
    method <- methods[method]
    if(anyNA(x) || (!is.null(y) && anyNA(y))) {
        stop("x and y must have no missing values")
    }
    q <- if(is.null(y)) ncol(x) else ncol(y)
    if(as.double(ncol(x)) * q >= .Machine$integer.max) {
        stop("too many pairs for one call; split the columns of y")
    }
    dn <- list(colnames(x), if(is.null(y)) colnames(x) else colnames(y))
    storage.mode(x) <- "double"
    if(!is.null(y)) storage.mode(y) <- "double"
    if(method == "spearman") {
        x <- .Call(C_gpuRank, x, precision)
        if(!is.null(y)) y <- .Call(C_gpuRank, y, precision)
    }

    # the permutations are drawn here, so set.seed() reproduces them
    perms <- vapply(seq_len(B), function(b) sample.int(n), integer(n)) - 1L
    dim(perms) <- c(n, B)
    # a permuted correlation equal to the observed one counts as at least
    # as extreme; this allows for rounding in the comparison
    eps <- if(precision == "double") 1e-10 else 1e-5
    ans <- .Call(C_gpuCorPerm, x, y, perms, eps, precision)

    r <- ans$r
    r[is.nan(r)] <- NA
    if(is.null(y)) diag(r) <- NA
    p <- (ans$counts + 1) / (B + 1)
    # Westfall and Young's single-step maxT: how often the permutation's
    # largest |r| over all the tests reaches this one
    maxima <- sort(ans$maxima)
    reach <- B - findInterval(abs(r) - eps, maxima, left.open = TRUE)
    adjusted <- (reach + 1) / (B + 1)
    p[is.na(r)] <- NA
    adjusted[is.na(r)] <- NA
    dim(adjusted) <- dim(r)
    if(!all(vapply(dn, is.null, NA))) {
        dimnames(r) <- dimnames(p) <- dimnames(adjusted) <- dn
    }
    list(coefficients = r, p.value = p, p.adjusted = adjusted,
         permutations = B)
}
