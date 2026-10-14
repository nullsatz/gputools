gpuMatMult <- function(a, b) {
	a <- as.matrix(a)
	b <- as.matrix(b)
	storage.mode(a) <- "double"
	storage.mode(b) <- "double"

	if (ncol(a) != nrow(b))
		stop("error:  matrix dimensions mismatched for matrix multiplication")
        
	.Call(C_gpuMatMult, a, b)
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
        
	results <- gpuMatMult(t(a), b)
	return(results)
}


gpuTcrossprod <- function(a, b=NULL) 
{
    a <- as.matrix(a)

    if (is.null(b)) b <- as.matrix(a)
    else b <- as.matrix(b)

    if (ncol(a) != ncol(b))
        stop("error: matrix dim mismatch for transposed cross-product")
        
	results <- gpuMatMult(a, t(b))
	return(results)
}
