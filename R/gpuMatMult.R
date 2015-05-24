gpuMatMult <- function(a, b) {
	a <- as.matrix(a)
	b <- as.matrix(b)

	if (ncol(a) != nrow(b))
		stop("error:  matrix dimensions mismatched for matrix multiplication")
        
	.Call("gpuMatMult", a, b, PACKAGE='gputools')
}

cpuMatMult <- function(a, b) {
	a <- as.matrix(a)
	b <- as.matrix(b)

	if (ncol(a) != nrow(b))
		stop("error:  matrix dimensions mismatched for matrix multiplication")
        
	a %*% b
}

gpuCrossprod <- function(a, b=NULL) {
    a <- as.matrix(a)

    if (is.null(b)) b <- as.matrix(a)
    else b <- as.matrix(b)

    if (nrow(a) != nrow(b))
        stop("error: matrix dim mismatch for cross-product.")
        
	results <- .Call("gpuMatMult", t(a), b, PACKAGE='gputools')
	return(results)
}


gpuTcrossprod <- function(a, b=NULL) 
{
    a <- as.matrix(a)

    if (is.null(b)) b <- as.matrix(a)
    else b <- as.matrix(b)

    if (ncol(a) != ncol(b))
        stop("error: matrix dim mismatch for transposed cross-product")
        
	results <- .Call("gpuMatMult", a,t(b), PACKAGE='gputools')
	return(results)
}
