gpuSolve <- function(x, y=NULL) {
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
		myCall <- .C("rGetInverseFromQR",
			as.integer(n), as.integer(p),
			as.single(x.q), as.single(x.r),
			inverse = single(n * p),
			PACKAGE='gputools'
		)
		x.inverse <- matrix(myCall$inverse, p, n)
		return(x.inverse)
	} else {
		y <- as.single(y)
		if(length(y) != n) {
			stop("y must have length nrows(x)")
		}
		myCall <- .C("rSolveFromQR",
			as.integer(n), as.integer(p),
			as.single(x.q), as.single(x.r),
			y, solution = single(p),
			PACKAGE='gputools'
		)
		return(myCall$solution)
	}
}
