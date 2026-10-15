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
	storage.mode(x) <- "double"

	# solve() names the solution's rows by x's columns
	if(is.null(y)) {
		b <- diag(n)
		sol <- .Call(C_gpuSolve, x, b, precision)
		dimnames(sol) <- rev(dimnames(x))
		return(sol)
	}

	isVector <- is.null(dim(y))
	b <- as.matrix(y)
	if(nrow(b) != n) {
		stop("y must have length nrows(x)")
	}
	storage.mode(b) <- "double"
	sol <- .Call(C_gpuSolve, x, b, precision)
	if(isVector) {
		sol <- drop(sol)
		names(sol) <- colnames(x)
	} else {
		dimnames(sol) <- list(colnames(x), colnames(b))
	}
	sol
}
