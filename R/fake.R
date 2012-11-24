gpuSvmTrain <- function(y, x, C = 10, kernelWidth = 0.125, eps = 0.5, 
	stoppingCrit = 0.001, isRegression = FALSE)
{
	if(TRUE) {
		stop("svm functions not implemented in device emulation mode")
	}

	# input
	m <- as.integer(nrow(x))
	n <- as.integer(ncol(x))

	y <- as.single(y)
	x <- as.single(x)

	C <- as.single(C)
	kernelWidth <- as.single(kernelWidth)
	eps <- as.single(eps)
	stoppingCrit <- as.single(stoppingCrit)

	regressionBit <- as.integer(0)
	if(isRegression) {
		regressionBit <- as.integer(1)
	}

	# output
	alpha <- single(m)
	beta <- single(1)
	numSvs <- integer(1)
	numPosSvs <- integer(1)

	if(isRegression) {
		call1 <- .C("R_SVRTrain", alpha = alpha, beta = beta,
			y, x,
			C, kernelWidth, eps, 
			m, n, 
			stoppingCrit, numSvs = numSvs)

		numSvs <- call1$numSvs
		numPosSvs <- integer(1)
		alpha <- call1$alpha
		beta <- call1$beta
					
		call2 <- .C("R_produceSupportVectors", regressionBit, m, n, 
			call1$numSvs, numPosSvs, x, y, call1$alpha,
			svCoefficients = single(numSvs),
			supportVectors = single(numSvs * n))
	} else {
		call1 <- .C("R_SVMTrain", alpha = alpha, beta = beta,
			y, x,
			C, kernelWidth,
			m, n, 
			stoppingCrit, numSvs = numSvs, numPosSvs = numPosSvs)
			
		numSvs <- call1$numSvs
		numPosSvs <- call1$numPosSvs
		alpha <- call1$alpha
		beta <- call1$beta

		call2 <- .C("R_produceSupportVectors", regressionBit, m, n,
			numSvs, numPosSvs, x, y, alpha,
			svCoefficients = single(numSvs),
			supportVectors = single(numSvs * n))
	}
	list(supportVectors = matrix(call2$supportVectors, numSvs, n),
		svCoefficients = call2$svCoefficients, svOffset = beta)
}

gpuSvmPredict <- function(data, supportVectors, svCoefficients, svOffset,
	kernelWidth = 0.125, isRegression = FALSE)
{
	if(TRUE) {
		stop("svm functions not implemented in device emulation mode")
	}

	m <- as.integer(nrow(data))
	k <- as.integer(ncol(data))
	n <- as.integer(length(svCoefficients))

	data <- as.single(data)
	supportVectors <- as.single(supportVectors)
	svCoefficients <- as.single(svCoefficients)
	svOffset <- as.single(svOffset)
	kernelWidth <- as.single(kernelWidth)

	predictions <- single(m)

	if(isRegression) {
		isRegression <- as.integer(1)
	} else {
		isRegression <- as.integer(0)
	}	

	classes <- .C("R_GPUPredictWrapper", m, n, k, 
		kernelWidth, data, supportVectors, svCoefficients,
		output = predictions,
		svOffset, isRegression)

	classes$output
}

getAucEstimate <- function(classes, scores)
{
	if(TRUE) {
		stop("svm functions not implemented in device emulation mode")
	}

	classes <- as.double(classes)
	n <- as.integer(length(classes))
	c <- .C("RgetAucEstimate", NAOK = TRUE, n, classes, scores, 
		auc = double(1))
	c$auc
}

gpuFastICA <-
function (X, n.comp, alg.typ = c("parallel","deflation"),
          fun = c("logcosh", "exp"),
          alpha = 1,
          row.norm = FALSE, maxit = 200, tol = 1e-04,
          verbose = FALSE, w.init=NULL)
{
	if("no" == "no") {
		stop("cula not found; gpuFastICA is disabled; use the fastICA package")
	}

    dd <- dim(X)
    d <- dd[dd != 1]
    if (length(d) != 2)
        stop("data must be matrix-conformal")
    X <- if (length(d) != length(dd))
        matrix(X, d[1], d[2])
    else as.matrix(X)

    if (alpha < 1 || alpha > 2)
        stop("alpha must be in range [1,2]")
    alg.typ <- match.arg(alg.typ)
    fun <- match.arg(fun)
    n <- nrow(X)
    p <- ncol(X)

    if (n.comp > min(n, p)) {
        message("'n.comp' is too large: reset to ", min(n, p))
        n.comp <- min(n, p)
    }
    if(is.null(w.init))
        w.init <- matrix(rnorm(n.comp^2),n.comp,n.comp)
    else {
        if(!is.matrix(w.init) || length(w.init) != (n.comp^2))
            stop("w.init is not a matrix or is the wrong size")
    }
	a <- .C("icainc_JM",
		as.single(X),
        as.single(w.init),
        as.integer(p),
        as.integer(n),
        as.integer(n.comp),
        as.single(alpha),
        as.integer(1),
        as.integer(row.norm),
        as.integer(1 + (fun == "exp")),
        as.integer(maxit),
        as.single(tol),
        as.integer(alg.typ != "parallel"),
        as.integer(verbose),
        X = single(p * n),
        K = single(n.comp * p),
        W = single(n.comp * n.comp),
        A = single(p * n.comp),
        S = single(n.comp * n),
		PACKAGE = 'gputools'
	)
	X1 <- t(matrix(a$X, p, n, byrow = TRUE))
	K <- t(matrix(a$K, n.comp, p, byrow = TRUE))
	W <- t(matrix(a$W, n.comp, n.comp, byrow = TRUE))
	A <- t(matrix(a$A, p, n.comp, byrow = TRUE))
	S <- t(matrix(a$S, n.comp, n, byrow = TRUE))
	return(list(X = X1, K = K, W = W, A = A, S = S))
}

gpuSvd <- function(x, nu = min(n,p), nv = min(n,p)) {
	if("no" == "no") {
		stop("cula not found; gpuSvd is disabled; use the svd function")
	}

    x <- as.matrix(x)
    if (any(!is.finite(x))) {
		stop("infinite or missing values in 'x'")
	}

    dx <- dim(x)
    n <- dx[1L]
    p <- dx[2L]

    if(!n || !p) {
		stop("0 extent dimensions")
	}
    if (is.complex(x)) {
		stop("complex arguments not yet supported by gpuSvd")
    }
    if(!is.numeric(x)) {
		stop("argument to 'svd' must be numeric")
	}

    size.diag <- min(n,p)

    if(nu == 0L) {
		jobu <- 'N'
		u <- single(0L)
    } else if(nu == n) {
		jobu <- 'A'
		u <- matrix(0, n, n)
		mode(u) <- 'single'
    } else if(nu == p) {
		jobu <- 'S'
		u <- matrix(0, n, size.diag)
		mode(u) <- 'single'
    } else {
		stop("'nu' must be 0, nrow(x) or ncol(x)")
	}

    if(nv == 0L) {
		jobv <- 'N'
		v <- single(0L)
    } else if(nv == p) {
		jobv <- 'A'
		v <- matrix(0, p, p)
		mode(v) <- 'single'
		nv <- p
    } else if(nv == n) {
		jobv <- 'S'
		v <- matrix(0, size.diag, p)
		mode(v) <- 'single'
		nv <- size.diag
    } else {
		stop("'nv' must be 0, nrow(x) or ncol(x)")
	}

    z <- .C("rSvd",
		jobu, jobv, as.integer(n), as.integer(p), as.single(x), as.integer(n),
		d = single(size.diag), u = u, as.integer(n), v = v, as.integer(nv),
		DUP=FALSE, PACKAGE='gputools'
	)[c("d","u","v","info")]

    if(nv && nv < p) {
		z$v <- z$v[, 1L:nv, drop = FALSE]
	}
	if(nv) {
		z$v <- t(z$v)
	}

    z[c("d", if(nu) "u", if(nv) "v")]
}
