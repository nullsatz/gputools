gpuCor <- function(x, y = NULL, use = "everything", method = "pearson",
                   precision = "single")
{
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
    useFlag <- 1L
    if (uses[use] == "pairwise.complete.obs") {
        useFlag <- 2L
    }


    precisions <- c("single", "double")
    precision <- pmatch(precision, precisions, -1)
    if(is.na(precision)) {
        stop("invalid correlation precision")
    }
    if(precision == -1) {
        stop("ambiguous correlation precision")
    }
    precisionFlag <- 2L
    if (precisions[precision] == "single") {
        precisionFlag <- 1L
    }

    a <- list()
    if(methods[method] == "pearson") {
        a <- pearson(useFlag, precisionFlag, x, y)
    } else if(methods[method] == "kendall") {
        if(uses[use] != "everything") {
            warning("NA handling for Kendall's is not yet supported.")
        }
        a <- kendall(x, y, precisionFlag)
        pairs <- matrix(size, nx, ny)
        a <- list(coefficients = a, pairs = pairs)
    } else {
        stop("This correlation method is not yet supported.")
    }
    return(a)
}

gpuTtest <- function(goodPairs, coeffs) {
    goodPairs <- as.single(goodPairs)
    coeffs <- as.single(coeffs)

    n <- as.integer(length(goodPairs))

    .C("rtestT", NAOK = TRUE,
       goodPairs, coeffs, n,
       results = single(n),
       PACKAGE = 'gputools')$results
}
