gpuSvd <- function(x, nu = min(n, p), nv = min(n, p),
	precision = getOption("gputools.precision", "single"))
{
	precision <- .precision(precision)
	x <- as.matrix(x)
	if(any(!is.finite(x))) {
		stop("infinite or missing values in 'x'")
	}
	dx <- dim(x)
	n <- dx[1L]
	p <- dx[2L]
	if(!n || !p) {
		stop("a dimension is zero")
	}
	k <- min(n, p)
	if(nu > k || nv > k) {
		stop("gpuSvd computes the thin decomposition: nu and nv cannot exceed min(nrow(x), ncol(x))")
	}
	storage.mode(x) <- "double"

	# cuSOLVER's SVD needs nrow >= ncol: factor a wide x as t(x), whose
	# left and right singular vectors are x's right and left ones
	wide <- n < p
	a <- if(wide) t(x) else x
	r <- .Call(C_gpuSvd, a, if(wide) nv > 0 else nu > 0,
		if(wide) nu > 0 else nv > 0, precision)
	u <- if(wide) r$v else r$u
	v <- if(wide) r$u else r$v

	res <- list(d = r$d)
	if(nu) {
		res$u <- u[, seq_len(nu), drop = FALSE]
	}
	if(nv) {
		res$v <- v[, seq_len(nv), drop = FALSE]
	}
	res
}

# prcomp.default, with the SVD on the GPU
gpuPrcomp <- function(x, retx = TRUE, center = TRUE, scale. = FALSE,
	tol = NULL, rank. = NULL,
	precision = getOption("gputools.precision", "single"))
{
	x <- as.matrix(x)
	x <- scale(x, center = center, scale = scale.)
	cen <- attr(x, "scaled:center")
	sc <- attr(x, "scaled:scale")
	if(any(sc == 0)) {
		stop("cannot rescale a constant/zero column to unit variance")
	}
	n <- nrow(x)
	p <- ncol(x)
	k <- if(!is.null(rank.)) {
		stopifnot(length(rank.) == 1, is.finite(rank.), as.integer(rank.) > 0)
		min(as.integer(rank.), n, p)
	} else {
		min(n, p)
	}
	s <- gpuSvd(x, nu = 0, nv = k, precision = precision)
	j <- seq_len(k)
	s$d <- s$d / sqrt(max(1, n - 1))
	if(!is.null(tol)) {
		rank <- sum(s$d > (s$d[1L] * tol))
		if(rank < k) {
			j <- seq_len(k <- rank)
			s$v <- s$v[, j, drop = FALSE]
		}
	}
	dimnames(s$v) <- list(colnames(x), paste0("PC", j))
	r <- list(sdev = s$d, rotation = s$v,
		center = if(is.null(cen)) FALSE else cen,
		scale = if(is.null(sc)) FALSE else sc)
	if(retx) {
		r$x <- x %*% s$v
	}
	class(r) <- "prcomp"
	r
}
