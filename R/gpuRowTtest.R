gpuRowTtest <- function(x, group, var.equal = FALSE, B = 0L,
    precision = getOption("gputools.precision", "single")) {
    precision <- .precision(precision)
    x <- as.matrix(x)
    storage.mode(x) <- "double"
    n <- ncol(x)
    if(length(group) != n) {
        stop("group must have one entry per column of x")
    }
    group <- as.factor(group)
    if(nlevels(group) != 2) {
        stop("group must have exactly two levels")
    }
    in1 <- group == levels(group)[1]
    n1 <- sum(in1)
    n2 <- n - n1
    if(n1 < 2 || n2 < 2) {
        stop("each group needs at least 2 samples")
    }
    if(anyNA(x) || anyNA(group)) {
        stop("x and group must have no missing values")
    }
    B <- as.integer(B)
    if(length(B) != 1 || is.na(B) || B < 0) {
        stop("B must be a non-negative whole number")
    }

    # the tests themselves, in double on the CPU: means, variances, t, df
    m1 <- rowMeans(x[, in1, drop = FALSE])
    m2 <- rowMeans(x[, !in1, drop = FALSE])
    v1 <- rowSums((x[, in1, drop = FALSE] - m1)^2) / (n1 - 1)
    v2 <- rowSums((x[, !in1, drop = FALSE] - m2)^2) / (n2 - 1)
    if(var.equal) {
        v <- ((n1 - 1) * v1 + (n2 - 1) * v2) / (n - 2)
        se <- sqrt(v * (1 / n1 + 1 / n2))
        df <- rep(n - 2, nrow(x))
    } else {
        a <- v1 / n1
        b <- v2 / n2
        se <- sqrt(a + b)
        df <- (a + b)^2 / (a^2 / (n1 - 1) + b^2 / (n2 - 1))
    }
    # a zero standard error with the means apart is a complete separation
    # of the groups: t infinite, p 0 (as SciPy reports it); with the means
    # equal as well, or a constant row, there is no test
    t <- (m1 - m2) / se
    t[se == 0 & m1 == m2] <- NaN
    p.value <- 2 * pt(-abs(t), df)
    # (Welch's df is 0/0 there, so the p-value is set directly)
    p.value[is.infinite(t)] <- 0
    if(!var.equal) df[is.infinite(t)] <- NA
    res <- data.frame(mean1 = m1, mean2 = m2, statistic = t, df = df,
                      p.value = p.value, row.names = rownames(x))

    if(B > 0) {
        # the permutations, drawn here so set.seed() reproduces them, on
        # the GPU: every row's t under every relabelling at once
        labels <- matrix(0L, n, B + 1L)
        labels[, 1] <- in1
        for(b in seq_len(B)) labels[, b + 1L] <- in1[sample.int(n)]
        if(n1 == n2) {
            # With equal groups a labelling and its swap give the same |t|
            # in exact arithmetic but are computed by different roundings,
            # which for strongly separated rows can differ by more than the
            # allowance: a swap of the observed labelling could then fail
            # to count, making p-values too small. Putting sample 1 in
            # group 1 always makes the two the same column, bit for bit.
            flip <- labels[1, ] == 0L
            labels[, flip] <- 1L - labels[, flip]
        }
        eps <- if(precision == "double") 1e-10 else 1e-5
        ans <- .Call(C_gpuRowTtest, x, labels, n1, isTRUE(var.equal), eps,
                     precision)
        p <- (ans$counts + 1) / (B + 1)
        maxima <- sort(ans$maxima)
        reach <- B - findInterval(abs(ans$t) * (1 - eps), maxima,
                                  left.open = TRUE)
        adjusted <- (reach + 1) / (B + 1)
        p[is.nan(t)] <- NA
        adjusted[is.nan(t)] <- NA
        res$p.perm <- p
        res$p.adjusted <- adjusted
    }
    res
}
