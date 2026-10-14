gpuQr <- function(x, tol = 1e-07) {

	x <- as.matrix(x)
	if(is.complex(x)) {
		stop("complex gpuQR not yet supported")
	}

	n <- nrow(x)
	p <- ncol(x)

	res <- .Call(C_gpuQrDecomp, x, as.double(tol))

        res$pivot <- res$pivot + 1L
        
	if(!is.null(cn <- colnames(x)))
		colnames(res$qr) <- cn[res$pivot]

	class(res) <- "qr"
	res
}
