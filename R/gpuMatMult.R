gpuMatMult <- function(a, b) 
{
	a <- as.matrix(a)
	b <- as.matrix(b)

	if (ncol(a) != nrow(b))
		stop("error:  matrix dimensions mismatched for matrix multiplication")
        
#	results <- .C("RgpuMatMult", as.integer(0L), as.integer(0L),
#		as.single(a), as.integer(nrow(a)), as.integer(ncol(a)),
#		as.single(b), as.integer(nrow(b)), as.integer(ncol(b)),
#		output = single(nrow(a)*ncol(b)),
#		PACKAGE='gputools')

	.Call("gpuMatMult", a, b)
}

gpuCrossprod <- function(a, b=NULL) 
{
	a <- as.matrix(a)
        if (is.null(b))
          b <- as.matrix(a)
        else
          b <- as.matrix(b)

        if (nrow(a) != nrow(b))
          stop("error:  matrix dimensions mismatched for cross-product.")
        
	results <- .C("RgpuMatMult", as.integer(1L), as.integer(0L),
		as.single(a), as.integer(nrow(a)), as.integer(ncol(a)),
		as.single(b), as.integer(nrow(b)), as.integer(ncol(b)),
		output = single(ncol(a)*ncol(b)),
		PACKAGE='gputools')

	matrix(results$output, ncol(a), ncol(b))
}


gpuTcrossprod <- function(a, b=NULL) 
{
	a <- as.matrix(a)
	b <- as.matrix(b)

        if (ncol(a) != ncol(b))
          stop("error:  matrix dimensions mismatched for transposed cross-product")
	results <- .C("RgpuMatMult", as.integer(0L), as.integer(1L),
		as.single(a), as.integer(nrow(a)), as.integer(ncol(a)),
		as.single(b), as.integer(nrow(b)), as.integer(ncol(b)),
		output = single(nrow(a)*nrow(b)),
		PACKAGE='gputools')

	matrix(results$output, nrow(a), nrow(b))
}
