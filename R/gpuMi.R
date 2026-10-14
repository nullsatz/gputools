gpuMi <- function(x, y = NULL, bins = 2, splineOrder = 1)
{
	x <- as.matrix(x)
	if(!is.null(y)) {
		y <- as.matrix(y)
	}

	bins <- as.integer(bins)
	splineOrder <- as.integer(splineOrder)

	nsamples <- as.integer(nrow(x))
	na <- as.integer(ncol(x))
	a <- as.double(x)

	b <- a
	nb <- na
	row_labels <- colnames(x)

	if(is.null(y)) {
		nb <- as.integer(ncol(x))
		b <- as.double(x)
		row_labels <- colnames(x)
	} else {
		nb <- as.integer(ncol(y))
		b <- as.double(y)
		row_labels <- colnames(y)
	}
        
	mi <- .Call(C_gpuMutualInfo, bins, splineOrder, nsamples, a, na, b, nb)
	mi <- matrix(mi, nb, na)
	rownames(mi) <- row_labels
	colnames(mi) <- colnames(x)
	return(mi)
}
