gpuCor <- function(x, y = NULL, use = "everything", method = "pearson",
    test = FALSE, precision = getOption("gputools.precision", "single")) {
    precision <- .precision(precision)
    x <- as.matrix(x)
    nx <- ncol(x)
    size <- nrow(x)

    if(!is.null(y)) {
        y <- as.matrix(y)
        if(nrow(y) != size) {
            stop("x and y have different numbers of rows")
        }
    }
    ny <- if(is.null(y)) nx else ncol(y)

    methods <- c("pearson", "kendall", "spearman")
    method <- pmatch(method, methods, -1)
    if(is.na(method)) {
        stop("invalid correlation method")
    }
    if(method == -1) {
        stop("ambiguous correlation method")
    }
    method <- methods[method]

    uses <- c("everything", "pairwise.complete.obs")
    use <- pmatch(use, uses, -1)
    if(is.na(use) || use == -1) {
        stop("use must be \"everything\" or \"pairwise.complete.obs\"")
    }
    use <- uses[use]
    storage.mode(x) <- "double"
    if(!is.null(y)) storage.mode(y) <- "double"
    dn <- list(colnames(x), colnames(if(is.null(y)) x else y))

    # a column with a missing value gives NA under use = "everything", as
    # with cor()
    incomplete <- function() outer(colSums(is.na(x)) > 0,
                                   colSums(is.na(y)) > 0, "|")

    selfCor <- is.null(y)
    if(selfCor) y <- x
    # no pair has to drop rows: every correlation uses all of them, and the
    # whole matrix is one product of standardised columns on the GPU
    allRows <- use == "everything" || !(anyNA(x) || anyNA(y))

    if(method == "pearson" ||
       (method == "spearman" && allRows)) {
        if(method == "spearman") {
            # cor()'s Spearman is Pearson's on the ranks (ties averaged)
            # (rank(ties.method = "average", na.last = "keep") per column,
            # on the GPU)
            x <- .Call(C_gpuRank, x, precision)
            if(!selfCor) y <- .Call(C_gpuRank, y, precision) else y <- x
        }
        if(allRows) {
            answer <- .Call(C_gpuCorProduct, x, if(selfCor) NULL else y,
                            precision)
            corr <- answer$corr
            ts <- answer$ts
            pairs <- matrix(as.double(size), nx, ny)
        } else {
            answer <- .Call(C_gpuPmcc, 1L, x, as.integer(nx), y,
                            as.integer(ny), as.integer(size), precision)
            pairs <- t(matrix(answer$pairs, ny, nx))
            corr <- t(matrix(answer$corr, ny, nx))
            ts <- .tScore(corr, pairs)
        }
    } else {
        ts <- NULL
        kernel <- if(method == "kendall") C_gpuKendall else C_gpuSpearman
        answer <- .Call(kernel, x, as.integer(nx), y, as.integer(ny),
                        as.integer(size), precision)
        corr <- matrix(answer$r, nx, ny)
        pairs <- matrix(answer$n, nx, ny)
        if(use == "everything") {
            corr[incomplete()] <- NA
            pairs[incomplete()] <- size
        }
    }
    corr[is.nan(corr)] <- NA
    dimnames(corr) <- dimnames(pairs) <- dn

    if(method == "kendall") {
        res <- list(coefficients = corr, pairs = pairs)
        if(test) {
            z <- .kendallZ(x, y, corr, pairs)
            dimnames(z) <- dn
            res$z <- z
            res$p.value <- 2 * pnorm(-abs(z))
        }
        return(res)
    }
    if(is.null(ts)) ts <- .tScore(corr, pairs)
    dimnames(ts) <- dn
    res <- list(coefficients = corr, ts = ts, pairs = pairs)
    if(test) {
        res$p.value <- 2 * pt(-abs(ts), pairs - 2)
    }
    res
}

# The t statistic of correlations r from n pairs, as cor.test() computes
# it: +-Inf at |r| = 1, NaN with fewer than three pairs.
.tScore <- function(r, n) {
    r[!is.na(r) & r > 1] <- 1
    r[!is.na(r) & r < -1] <- -1
    t <- r * sqrt((n - 2) / (1 - r^2))
    t[!is.na(r) & n < 3] <- NaN
    t
}

# Kendall's z with cor.test(method = "kendall", exact = FALSE)'s tie
# correction: S = tau_b sqrt((n0 - Tx)(n0 - Ty)) over its variance under
# independence, from each pair's complete rows.
.kendallZ <- function(x, y, tau, n) {
    tieStats <- function(v) {
        t <- table(v)
        t <- as.numeric(t[t > 1])
        c(t1 = sum(t * (t - 1)), t2 = sum(t * (t - 1) * (2 * t + 5)),
          t3 = sum(t * (t - 1) * (t - 2)))
    }
    zOf <- function(tau, m, a1, a2, a3, b1, b2, b3) {
        n0 <- m * (m - 1) / 2
        S <- tau * sqrt((n0 - a1 / 2) * (n0 - b1 / 2))
        v <- (m * (m - 1) * (2 * m + 5) - a2 - b2) / 18 +
            a1 * b1 / (2 * m * (m - 1)) + a3 * b3 / (9 * m * (m - 1) * (m - 2))
        S / sqrt(v)
    }
    # columns without missing values: their own tie counts, all pairs at once
    sx <- vapply(seq_len(ncol(x)), function(i) tieStats(x[, i]), numeric(3))
    sy <- vapply(seq_len(ncol(y)), function(j) tieStats(y[, j]), numeric(3))
    a <- function(k) matrix(sx[k, ], ncol(x), ncol(y))
    b <- function(k) matrix(sy[k, ], ncol(x), ncol(y), byrow = TRUE)
    z <- zOf(tau, n, a(1), a(2), a(3), b(1), b(2), b(3))
    # pairs with a missing value: ties among that pair's complete rows
    naX <- colSums(is.na(x)) > 0
    naY <- colSums(is.na(y)) > 0
    todo <- which(outer(naX, naY, "|") & !is.na(tau), arr.ind = TRUE)
    for(r in seq_len(nrow(todo))) {
        i <- todo[r, 1]
        j <- todo[r, 2]
        ok <- !(is.na(x[, i]) | is.na(y[, j]))
        ta <- tieStats(x[ok, i])
        tb <- tieStats(y[ok, j])
        z[i, j] <- zOf(tau[i, j], n[i, j], ta[[1]], ta[[2]], ta[[3]],
                       tb[[1]], tb[[2]], tb[[3]])
    }
    z
}

gpuTtest <- function(goodPairs, coeffs,
    precision = getOption("gputools.precision", "single")) {
    precision <- .precision(precision)
    .Call(C_gpuTtest, as.double(goodPairs), as.double(coeffs), precision)
}
