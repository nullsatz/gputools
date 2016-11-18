gpuQr <- function(x, tol = 1e-07) {

	x <- as.matrix(x)
	if(is.complex(x)) {
		stop("complex gpuQR not yet supported")
	}

	n <- nrow(x)
	p <- ncol(x)

	mode(x) <- 'single'

	res <- .C("rGetQRDecompRR",
		as.integer(n),
		as.integer(p),
		as.double(tol),
		qr = x,
		pivot = as.integer(0L:(p-1)),
		qraux = double(p),
		rank = integer(1L),
		PACKAGE='gputools'
	)[c('qr', 'pivot', 'qraux', 'rank')]

        res$pivot <- res$pivot + 1
        
	if(!is.null(cn <- colnames(x)))
		colnames(res$qr) <- cn[res$pivot]

	class(res) <- "qr"
	res
}
