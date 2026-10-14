gpuSolve <- function(x, y=NULL,
	precision = getOption("gputools.precision", "single")) {
	precision <- .precision(precision)
	x <- as.matrix(x)
	if(is.complex(x)) {
		stop("complex gpuSolve not yet supported")
	}

	n <- nrow(x)
	p <- ncol(x)
	if(p > n) {
		stop("x represents an underdetermined system")
	}

	x.qr <- qr(x)
	x.q <- qr.Q(x.qr)
	x.r <- qr.R(x.qr)

	if(is.null(y)) {
		inverse <- .Call(C_gpuInverseFromQR, x.q, x.r,
			as.integer(n), as.integer(p), precision)
		x.inverse <- matrix(inverse, p, n)
		return(x.inverse)
	} else {
		y <- as.double(y)
		if(length(y) != n) {
			stop("y must have length nrows(x)")
		}
		return(.Call(C_gpuSolveFromQR, x.q, x.r, y,
			as.integer(n), as.integer(p), precision))
	}
}
