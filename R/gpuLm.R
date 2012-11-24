#  File R/gpuLm.R
#  a substantial portion of this code is taken from
#  lm.R, lsfit.R and glm.R which are parts of the R base package,
#  http://www.R-project.org
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  A copy of the GNU General Public License is available at
#  http://www.r-project.org/Licenses/

gpuLm <- function (formula, data, subset, weights, na.action,
		method = "qr", model = TRUE, x = FALSE, y = FALSE,
		qr = TRUE, singular.ok = TRUE, contrasts = NULL,
		useSingle = TRUE, offset, ...)
{
	ret.x <- x
	ret.y <- y

	cl <- match.call()

	mf <- match.call(expand.dots = FALSE)
	m <- match(c("formula", "data", "subset", "weights", "na.action", "offset"),
		   names(mf), 0L)
	mf <- mf[c(1L, m)]
	mf$drop.unused.levels <- TRUE
	mf[[1L]] <- as.name("model.frame")
	mf <- eval(mf, parent.frame())

	if(useSingle != TRUE) {
		stop("Double precision gpuLm not yet implemented.\n\tSorry for any inconvenience!")
	}

	if (method == "model.frame") {
		return(mf)
	} else if (method != "qr") {
		warning(gettextf("method = '%s' is not supported. Using 'qr'",
			method), domain = NA)
	}

	mt <- attr(mf, "terms") # allow model.frame to update it
	y <- model.response(mf, "numeric")
	## avoid any problems with 1D or nx1 arrays by as.vector.
	w <- as.vector(model.weights(mf))
	if(!is.null(w) && !is.numeric(w)) {
		stop("'weights' must be a numeric vector")
	}
	offset <- as.vector(model.offset(mf))
	if(!is.null(offset)) {
		if(length(offset) != NROW(y)) {
			stop(gettextf("number of offsets is %d, should equal %d (number of observations)", length(offset), NROW(y)), domain = NA)
		}
	}

	if (is.empty.model(mt)) {
		x <- NULL
		z <- list(coefficients = if (is.matrix(y))
			matrix(,0,3) else numeric(0L), residuals = y,
			fitted.values = 0 * y, weights = w, rank = 0L,
		 	df.residual = if (is.matrix(y)) nrow(y) else length(y))
		if(!is.null(offset)) {
			z$fitted.values <- offset
			z$residuals <- y - offset
		}
	} else {
		x <- model.matrix(mt, mf, contrasts)
		if(is.null(w)) {
			z <- gpuLm.fit(x, y, NULL, offset = offset, useSingle = useSingle,
				singular.ok=singular.ok, ...)
		} else {
			z <- gpuLm.fit(x, y, w, offset = offset, useSingle = useSingle,
				singular.ok=singular.ok, ...)
		}
	}

	class(z) <- c(if(is.matrix(y)) 'mlm', 'gpuLm', 'lm')

	z$na.action <- attr(mf, 'na.action')
	z$offset <- offset
	z$contrasts <- attr(x, 'contrasts')
	z$xlevels <- .getXlevels(mt, mf)
	z$call <- cl
	z$terms <- mt

	if (model) {
		z$model <- mf
	}
	if (ret.x) {
		z$x <- x
	}
	if (ret.y) {
		z$y <- y
	}
	z
}

## Computes a default tolerance based on the precision of the data.
#
gpuLm.defaultTol <- function(useSingle = TRUE) {
	if (useSingle) {
		tol <- 1e-04
	} else {
		tol <- 1e-07
	}
	tol
}

gpuLm.fit <- function (x, y, w = NULL, offset = NULL, method = "qr",
	useSingle = TRUE, tol = gpuLm.defaultTol(useSingle), singular.ok = TRUE,
	...)
{
	if(useSingle != TRUE) {
		stop("Double precision gpuLm not yet implemented.\n\tSorry for any inconvenience!")
	}
	if (is.null(n <- nrow(x))) {
		stop("'x' must be a matrix")
	}
	n <- nrow(x)
	if(n == 0L) {
		stop("0 (non-NA) cases")
	}
	
	p <- ncol(x)
	if (p == 0L) { ## oops, null model
		return(list(coefficients = numeric(0L), residuals = y,
			fitted.values = 0 * y, weights = w, rank = 0,
			df.residual = length(y)))
	}
	
	ny <- NCOL(y)
	## treat one-col matrix as vector
	if(is.matrix(y) && (ny == 1)) {
		y <- drop(y)
	}
	if(!is.null(offset)) {
		y <- y - offset
	}
	if (NROW(y) != n) {
		stop("incompatible dimensions")
	}
	
	if(method != "qr") {
		warning(gettextf("method = '%s' is not supported. Using 'qr'",
			method), domain = NA)
	}
	if(length(list(...))) {
		warning("extra arguments ", paste(names(list(...)), sep=", "),
			" are just disregarded.")
	}

	x.asgn <- attr(x, "assign")
	if(!is.null(w)){
		if(length(w) != n) {
			stop("incompatible dimensions")
		}
		if(any(w < 0 | is.na(w))) {
			stop("missing or negative weights not allowed")
		}

		zero.weights <- any(w == 0)
		if(zero.weights) {
			save.r <- y
			save.f <- y
			save.w <- w
			ok <- w != 0
			nok <- !ok
			w <- w[ok]
			x0 <- x[!ok, , drop = FALSE]
			x <- x[ok,  , drop = FALSE]
			n <- nrow(x)
			y0 <- if (ny > 1L) y[!ok, , drop = FALSE] else y[!ok]
			y  <- if (ny > 1L) y[ ok, , drop = FALSE] else y[ok]
		}

		wts <- sqrt(w)
	}

	## Ensures that coefficients and residuals are initialized to zero
	## and that all persistent variables are passed in as floats.

	qraux <- double(p)
	coef <- mat.or.vec(p,ny)
	resid <- mat.or.vec(n,ny)

	if(is.null(w)) {
		qr <- x
		yIn <- y
	} else {
		qr <- x * wts
		yIn <- y * wts
	}

	mode(qr) <- "single"
	mode(yIn) <- "single"
	mode(coef) <- "single"
	mode(resid) <- "single"

	z <- .C("RgpuLSFit", PACKAGE="gputools",
		qr = qr, n = as.integer(n), p = as.integer(p),
		yIn, ny = as.integer(ny),
		tol = as.double(tol),
		coefficients = coef, residuals = resid,
		effects = yIn,
		rank = integer(1L), pivot = as.integer(0L:(p-1)),
		qraux = qraux, useSingle)[c('qr', 'n', 'p', 'ny', 'tol',
			'coefficients', 'residuals', 'effects', 'rank', 'pivot',
			'qraux')]

    # Removes "Csingle" attribute, to prevent dangerous side-effects.
	attr(qr, "Csingle") <- NULL
	attr(yIn, "Csingle") <- NULL
	attr(z$qr, "Csingle") <- NULL
	attr(z$coefficients, "Csingle") <- NULL
	attr(z$residuals, "Csingle") <- NULL
	attr(z$effects, "Csingle") <- NULL

	if(!singular.ok && z$rank < p) {
		stop("singular fit encountered")
	}
	
	coef <- z$coefficients

	# Compensates for C's zero-based indexing:
	z$pivot <- z$pivot + 1
	pivot <- z$pivot
	
	## careful here: the rank might be 0
	r1 <- seq_len(z$rank)
	
	dn <- colnames(x)
	if(is.null(dn)) {
		dn <- paste("x", 1L:p, sep="")
	}
	
	nmeffects <- c(dn[r1], rep.int("", n - z$rank))
	r2 <- if(z$rank < p) (z$rank+1L):p else integer(0L)
	
	if (is.matrix(y)) {
		coef[r2, ] <- NA
		coef[pivot, ] <- coef
		dimnames(coef) <- list(dn, colnames(y))
		dimnames(z$effects) <- list(nmeffects, colnames(y))
                dimnames(z$residuals) <- dimnames(y)
	} else {
		coef[r2] <- NA
		coef[pivot] <- coef
		names(coef) <- dn
		names(z$effects) <- nmeffects
                names(z$residuals) <- names(y)
	}
	
	z$coefficients <- coef
	
	r1 <- y - z$residuals
	if(!is.null(offset)) {
		r1 <- r1 + offset
	}
		
	# make a fake qr object to help R's base plot functions	
	qr <- z[c("qr", "qraux", "pivot", "tol", "rank")] 
	colnames(qr$qr) <- colnames(x)[qr$pivot]
	
	result <- c(
		z[c("coefficients", "residuals", "effects", "rank")],
		list(fitted.values = r1, assign = x.asgn,
			qr = structure(qr, class = 'qr'), df.residual = n - z$rank)
	)

	if(!is.null(w)){
		result$residuals <- result$residuals / wts
		result$fitted.values <- y - result$residuals
		result$weights <- w

		if(zero.weights) {
			coef[is.na(coef)] <- 0
			f0 <- x0 %*% coef

			if (ny > 1) {
				save.r[ok, ] <- result$residuals
				save.r[nok, ] <- y0 - f0
				save.f[ok, ] <- result$fitted.values
				save.f[nok, ] <- f0
			} else {
				save.r[ok] <- result$residuals
				save.r[nok] <- y0 - f0
				save.f[ok] <- result$fitted.values
				save.f[nok] <- f0
			}

			result$residuals <- save.r
			result$fitted.values <- save.f
			result$weights <- save.w
		}
	}
        rm(yIn)
        rm(qr)
        rm(coef)
        rm(resid)
	return(result)
}

## Taken almost verbatim from "stats" package lsfit() command.
#
gpuLsfit <- function(x, y, wt=NULL, intercept=TRUE, useSingle = TRUE,
	tolerance=gpuLm.defaultTol(useSingle), yname=NULL)
{
    ## find names of x variables (design matrix)

    x <- as.matrix(x)
    y <- as.matrix(y)
    xnames <- colnames(x)
    if( is.null(xnames) ) {
	if(ncol(x)==1) xnames <- "X"
	else xnames <- paste("X", 1L:ncol(x), sep="")
    }
    if( intercept ) {
	x <- cbind(1, x)
	xnames <- c("Intercept", xnames)
    }

    ## find names of y variables (responses)

    if(is.null(yname) && ncol(y) > 1) yname <- paste("Y", 1L:ncol(y), sep="")

    ## remove missing values

    good <- complete.cases(x, y, wt)
    dimy <- dim(as.matrix(y))
    if( any(!good) ) {
	warning(gettextf("%d missing values deleted", sum(!good)), domain = NA)
	x <- as.matrix(x)[good, ]
	y <- as.matrix(y)[good, ]
	wt <- wt[good]
    }

    ## check for compatible lengths

    nrx <- NROW(x)
    ncx <- NCOL(x)
    nry <- NROW(y)
    ncy <- NCOL(y)
    nwts <- length(wt)
    if(nry != nrx)
        stop(gettextf("'X' matrix has %d responses, 'Y' has %d responses",
                      nrx, nry), domain = NA)
    if(nry < ncx)
        stop(gettextf("%d responses, but only %d variables", nry, ncx),
             domain = NA)

    ## check weights if necessary

    if( !is.null(wt) ) {
	if(any(wt < 0)) stop("negative weights not allowed")
	if(nwts != nry)
            stop(gettextf("number of weights = %d should equal %d (number of responses)", nwts, nry), domain = NA)
	wtmult <- wt^0.5
	if( any(wt==0) ) {
	    xzero <- as.matrix(x)[wt==0, ]
	    yzero <- as.matrix(y)[wt==0, ]
	}
	x <- x*wtmult
	y <- y*wtmult
	invmult <- 1/ifelse(wt==0, 1, wtmult)
    }

    ## calls coprocessor interface

    storage.mode(x) <- "double"
    storage.mode(y) <- "double"

    ## Ensures that coefficients and residuals are initialized to zero
    ## and that all persistent variables are passed in as floats.

    qraux <- double(ncx)
    coef <- mat.or.vec(ncx,ncy)
    resid <- mat.or.vec(nrx,ncy)
    qr <- x
    yIn <- y

    mode(qr) <- "single"
    mode(yIn) <- "single"
    mode(coef) <- "single"
    mode(resid) <- "single"

    z <- .C("RgpuLSFit", PACKAGE="gputools",
		qr = qr, n = as.integer(nrx), p = as.integer(ncx),
		yIn, ny = as.integer(ncy),
		tol = as.double(tolerance),
		coefficients = coef, residuals = resid,
		effects = drop(yIn),
		rank = integer(1L), pivot = as.integer(0L:(ncx-1)),
		qraux = qraux, useSingle)[c('qr', 'n', 'p', 'ny', 'tol',
			'coefficients', 'residuals', 'effects', 'rank', 'pivot',
			'qraux')]

    # Removes "Csingle" attribute, to prevent dangerous side-effects.
    attr(qr, "Csingle") <- NULL
    attr(yIn, "Csingle") <- NULL
    attr(z$qr, "Csingle") <- NULL
    attr(z$coefficients, "Csingle") <- NULL
    attr(z$residuals, "Csingle") <- NULL
    attr(z$effects, "Csingle") <- NULL

    # Compensates for C's zero-based indexing.
    #
    z$pivot <- z$pivot + 1
    
    ## dimension and name output from call return.

    resids <- array(NA, dim=dimy)
    dim(z$residuals) <- c(nry, ncy)
    if(!is.null(wt)) {
	if(any(wt==0)) {
	    if(ncx==1) fitted.zeros <- xzero * z$coefficients
	    else fitted.zeros <- xzero %*% z$coefficients
	    z$residuals[wt==0, ] <- yzero - fitted.zeros
	}
	z$residuals <- z$residuals*invmult
    }
    resids[good, ] <- z$residuals
    if(dimy[2L] == 1 && is.null(yname)) {
	resids <- as.vector(resids)
	names(z$coefficients) <- xnames
    }
    else {
	colnames(resids) <- yname
	colnames(z$effects) <- yname
	dim(z$coefficients) <- c(ncx, ncy)
	dimnames(z$coefficients) <- list(xnames, yname)
    }
    z$qr <- as.matrix(z$qr)
    colnames(z$qr) <- xnames
    output <- list(coefficients=z$coefficients, residuals=resids)

    ## if X matrix was collinear, then the columns would have been
    ## pivoted hence xnames need to be corrected

    if( z$rank != ncx ) {
	xnames <- xnames[z$pivot]
	dimnames(z$qr) <- list(NULL, xnames)
	warning("'X' matrix was collinear")
    }

    ## return weights if necessary

    if (!is.null(wt) ) {
	weights <- rep.int(NA, dimy[1L])
	weights[good] <- wt
	output <- c(output, list(wt=weights))
    }

    ## return rest of output

    rqr <- list(qt=z$effects, qr=z$qr, qraux=z$qraux, rank=z$rank,
		pivot=z$pivot, tol=z$tol)
    oldClass(rqr) <- "qr"
    output <- c(output, list(intercept=intercept, qr=rqr))
    return(output)
}

## These functions are reproduced nearly verbatim from their
## counterparts in the "stats" package.
#
gpuGlm <- function(formula, family = gaussian, data, weights,
		subset, na.action, start = NULL,
		etastart, mustart, offset,
                useSingle = TRUE,
		control = gpuGlm.control(useSingle, ...), model = TRUE,
                method = "gpuGlm.fit", x = FALSE, y = TRUE,
                contrasts = NULL, ...)
{
    call <- match.call()
    ## family
    if(is.character(family))
        family <- get(family, mode = "function", envir = parent.frame())
    if(is.function(family)) family <- family()
    if(is.null(family$family)) {
	stop("'family' not recognized")
    }
    if(useSingle != TRUE) {
	stop("Double precision gpuLm not yet implemented.\n\tSorry for any inconvenience!")
    }


    ## extract x, y, etc from the model formula and frame
    if(missing(data)) data <- environment(formula)
    mf <- match.call(expand.dots = FALSE)
    m <- match(c("formula", "data", "subset", "weights", "na.action",
                 "etastart", "mustart", "offset"), names(mf), 0L)
    mf <- mf[c(1, m)]
    mf$drop.unused.levels <- TRUE
    mf[[1L]] <- as.name("model.frame")
    mf <- eval(mf, parent.frame())
    if(identical(method, "model.frame")) return(mf)
    ## we use 'glm.fit' for the sake of error messages.
    if(identical(method, "glm.fit")) {
        ## OK
    } else if(is.function(method)) {
        glm.fit <- method
    } else if(is.character(method)) {
        if(exists(method)) glm.fit <- get(method)
        else stop(gettextf("invalid 'method': %s", method), domain = NA)
    } else stop("invalid 'method'")
    mt <- attr(mf, "terms") # allow model.frame to have updated it

    Y <- model.response(mf, "any") # e.g. factors are allowed
    ## avoid problems with 1D arrays, but keep names
    if(length(dim(Y)) == 1L) {
        nm <- rownames(Y)
        dim(Y) <- NULL
        if(!is.null(nm)) names(Y) <- nm
    }
    ## null model support
    X <- if (!is.empty.model(mt)) model.matrix(mt, mf, contrasts) else matrix(,NROW(Y), 0L)
    ## avoid any problems with 1D or nx1 arrays by as.vector.
    weights <- as.vector(model.weights(mf))
    if(!is.null(weights) && !is.numeric(weights))
        stop("'weights' must be a numeric vector")
    ## check weights and offset
    if( !is.null(weights) && any(weights < 0) )
	stop("negative weights not allowed")
    offset <- as.vector(model.offset(mf))
    if(!is.null(offset)) {
        if(length(offset) != NROW(Y))
            stop(gettextf("number of offsets is %d should equal %d (number of observations)", length(offset), NROW(Y)), domain = NA)
    }
    ## these allow starting values to be expressed in terms of other vars.
    mustart <- model.extract(mf, "mustart")
    etastart <- model.extract(mf, "etastart")

    ## fit model via iterative reweighted least squares
    fit <- gpuGlm.fit(x = X, y = Y, weights = weights, start = start,
                   etastart = etastart, mustart = mustart,
                   offset = offset, family = family, useSingle = useSingle, control = control,
                   intercept = attr(mt, "intercept") > 0)

    ## This calculated the null deviance from the intercept-only model
    ## if there is one, otherwise from the offset-only model.
    ## We need to recalculate by a proper fit if there is intercept and
    ## offset.
    ##
    ## The gpuGlm.fit calculation could be wrong if the link depends on the
    ## observations, so we allow the null deviance to be forced to be
    ## re-calculated by setting an offset (provided there is an intercept).
    ## Prior to 2.4.0 this was only done for non-zero offsets.
    if(length(offset) && attr(mt, "intercept") > 0L) {
	fit$null.deviance <-
	    gpuGlm.fit(x = X[,"(Intercept)",drop=FALSE], y = Y, weights = weights,
                    offset = offset, family = family, useSingle = useSingle,
                    control = control, intercept = TRUE)$deviance
    }
    if(model) fit$model <- mf
    fit$na.action <- attr(mf, "na.action")
    if(x) fit$x <- X
    if(!y) fit$y <- NULL
    fit <- c(fit, list(call = call, formula = formula,
		       terms = mt, data = data,
		       offset = offset, control = control, method = method,
		       contrasts = attr(X, "contrasts"),
                       xlevels = .getXlevels(mt, mf)))
    class(fit) <- c("glm", "lm")
    fit
}

gpuGlm.defaultEps <- function(useSingle) {
  if (useSingle) {
    eps <- 1e-05
  }
  else {
    eps <- 1e-08
  }
  eps
}

gpuGlm.control <- function(useSingle, epsilon = gpuGlm.defaultEps(useSingle), maxit = 25, trace = FALSE)
{
    if(!is.numeric(epsilon) || epsilon <= 0)
	stop("value of 'epsilon' must be > 0")
    if(!is.numeric(maxit) || maxit <= 0)
	stop("maximum number of iterations must be > 0")
    list(epsilon = epsilon, maxit = maxit, trace = trace)
}

## Modified by Thomas Lumley 26 Apr 97
## Added boundary checks and step halving
## Modified detection of fitted 0/1 in binomial
## Updated by KH as suggested by BDR on 1998/06/16

gpuGlm.fit <-
    function (x, y, weights = rep(1, nobs), start = NULL,
	      etastart = NULL, mustart = NULL, offset = rep(0, nobs),
	      family = gaussian(), useSingle = TRUE, control = gpuGlm.control(useSingle), intercept = TRUE)
{
    x <- as.matrix(x)
    xnames <- dimnames(x)[[2L]]
    ynames <- if(is.matrix(y)) rownames(y) else names(y)
    conv <- FALSE
    nobs <- NROW(y)
    nvars <- ncol(x)
    EMPTY <- nvars == 0
    ## define weights and offset if needed
    if (is.null(weights))
	weights <- rep.int(1, nobs)
    if (is.null(offset))
	offset <- rep.int(0, nobs)

    ## get family functions:
    variance <- family$variance
    linkinv  <- family$linkinv
    if (!is.function(variance) || !is.function(linkinv) )
	stop("'family' argument seems not to be a valid family object")
    dev.resids <- family$dev.resids
    aic <- family$aic
    mu.eta <- family$mu.eta
    unless.null <- function(x, if.null) if(is.null(x)) if.null else x
    valideta <- unless.null(family$valideta, function(eta) TRUE)
    validmu  <- unless.null(family$validmu,  function(mu) TRUE)
    if(is.null(mustart)) {
        ## calculates mustart and may change y and weights and set n (!)
        eval(family$initialize)
    } else {
        mukeep <- mustart
        eval(family$initialize)
        mustart <- mukeep
    }
    if(EMPTY) {
        eta <- rep.int(0, nobs) + offset
        if (!valideta(eta))
            stop("invalid linear predictor values in empty model")
        mu <- linkinv(eta)
        ## calculate initial deviance and coefficient
        if (!validmu(mu))
            stop("invalid fitted means in empty model")
        dev <- sum(dev.resids(y, mu, weights))
        w <- ((weights * mu.eta(eta)^2)/variance(mu))^0.5
        residuals <- (y - mu)/mu.eta(eta)
        good <- rep(TRUE, length(residuals))
        boundary <- conv <- TRUE
        coef <- numeric(0L)
        iter <- 0L
    } else {
        coefold <- NULL
        eta <-
            if(!is.null(etastart)) etastart
            else if(!is.null(start))
                if (length(start) != nvars)
                    stop(gettextf("length of 'start' should equal %d and correspond to initial coefs for %s", nvars, paste(deparse(xnames), collapse=", ")),
                         domain = NA)
                else {
                    coefold <- start
                    offset + as.vector(if (NCOL(x) == 1) x * start else x %*% start)
                }
            else family$linkfun(mustart)
        mu <- linkinv(eta)
        if (!(validmu(mu) && valideta(eta)))
            stop("cannot find valid starting values: please specify some")
        ## calculate initial deviance and coefficient
        devold <- sum(dev.resids(y, mu, weights))
        boundary <- conv <- FALSE

        ##------------- THE Iteratively Reweighting L.S. iteration -----------
        for (iter in 1L:control$maxit) {
            good <- weights > 0
            varmu <- variance(mu)[good]
            if (any(is.na(varmu)))
                stop("NAs in V(mu)")
            if (any(varmu == 0))
                stop("0s in V(mu)")
            mu.eta.val <- mu.eta(eta)
            if (any(is.na(mu.eta.val[good])))
                stop("NAs in d(mu)/d(eta)")
            ## drop observations for which w will be zero
            good <- (weights > 0) & (mu.eta.val != 0)

            if (all(!good)) {
                conv <- FALSE
                warning("no observations informative at iteration ", iter)
                break
            }
            z <- (eta - offset)[good] + (y - mu)[good]/mu.eta.val[good]
            w <- sqrt((weights[good] * mu.eta.val[good]^2)/variance(mu)[good])
            ngoodobs <- as.integer(nobs - sum(!good))


	## Ensures that coefficients and residuals are initialized to zero
	## and that all persistent variables are passed in as floats.

	qraux <- double(nvars)
	coef <- double(nvars)
	resid <- double(ngoodobs)

	qr <- x[good, ] * w
	yIn <- w * z
        effects <- yIn


	mode(qr) <- "single"
	mode(yIn) <- "single"
	mode(coef) <- "single"
	mode(resid) <- "single"
        mode(effects) <- "single"

	fit <- .C("RgpuLSFit", PACKAGE="gputools",
		qr = qr, n = as.integer(ngoodobs), p = as.integer(nvars),
		yIn, ny = as.integer(1L),
		tol = as.double(min(gpuLm.defaultTol(useSingle), control$epsilon/1000)),
		coefficients = coef, residuals = resid,
		effects = effects,
		rank = integer(1L), pivot = as.integer(0L:(nvars-1)),
		qraux = qraux, useSingle=useSingle)[c('qr', 'n', 'p', 'ny', 'tol',
			'coefficients', 'residuals', 'effects', 'rank', 'pivot',
			'qraux')]

    # Removes "Csingle" attribute, to prevent dangerous side-effects.
	attr(qr, "Csingle") <- NULL
	attr(yIn, "Csingle") <- NULL
	attr(fit$qr, "Csingle") <- NULL
	attr(fit$coefficients, "Csingle") <- NULL
	attr(fit$residuals, "Csingle") <- NULL
	attr(fit$effects, "Csingle") <- NULL

        fit$pivot <- fit$pivot + 1
            
            if (any(!is.finite(fit$coefficients))) {
                conv <- FALSE
                warning("non-finite coefficients at iteration ", iter)
                break
            }
            ## stop if not enough parameters
            if (nobs < fit$rank)
                stop(gettextf("X matrix has rank %d, but only %d observations",
                              fit$rank, nobs), domain = NA)
            ## calculate updated values of eta and mu with the new coef:
            start[fit$pivot] <- fit$coefficients
            eta <- drop(x %*% start)
            mu <- linkinv(eta <- eta + offset)
            dev <- sum(dev.resids(y, mu, weights))

            if (control$trace)
                cat("Deviance =", dev, "Iterations -", iter, "\n")
            ## check for divergence
            boundary <- FALSE
            if (!is.finite(dev)) {
                if(is.null(coefold))
                    stop("no valid set of coefficients has been found: please supply starting values", call. = FALSE)
                warning("step size truncated due to divergence", call. = FALSE)
                ii <- 1
                while (!is.finite(dev)) {
                    if (ii > control$maxit)
                        stop("inner loop 1; cannot correct step size")
                    ii <- ii + 1
                    start <- (start + coefold)/2
                    eta <- drop(x %*% start)
                    mu <- linkinv(eta <- eta + offset)
                    dev <- sum(dev.resids(y, mu, weights))
                }
                boundary <- TRUE
                if (control$trace)
                    cat("Step halved: new deviance =", dev, "\n")
            }
            ## check for fitted values outside domain.
            if (!(valideta(eta) && validmu(mu))) {
                if(is.null(coefold))
                    stop("no valid set of coefficients has been found: please supply starting values", call. = FALSE)
                warning("step size truncated: out of bounds", call. = FALSE)
                ii <- 1
                while (!(valideta(eta) && validmu(mu))) {
                    if (ii > control$maxit)
                        stop("inner loop 2; cannot correct step size")
                    ii <- ii + 1
                    start <- (start + coefold)/2
                    eta <- drop(x %*% start)
                    mu <- linkinv(eta <- eta + offset)
                }
                boundary <- TRUE
                dev <- sum(dev.resids(y, mu, weights))
                if (control$trace)
                    cat("Step halved: new deviance =", dev, "\n")
            }
            ## check for convergence
            if (abs(dev - devold)/(0.1 + abs(dev)) < control$epsilon) {
                conv <- TRUE
                coef <- start
                break
            } else {
                devold <- dev
                coef <- coefold <- start
            }
        } ##-------------- end IRLS iteration -------------------------------

        if (!conv) warning("algorithm did not converge")
        if (boundary) warning("algorithm stopped at boundary value")
        eps <- 10*.Machine$double.eps
        if (family$family == "binomial") {
            if (any(mu > 1 - eps) || any(mu < eps))
                warning("fitted probabilities numerically 0 or 1 occurred")
        }
        if (family$family == "poisson") {
            if (any(mu < eps))
                warning("fitted rates numerically 0 occurred")
        }
        ## If X matrix was not full rank then columns were pivoted,
        ## hence we need to re-label the names ...
        ## Original code changed as suggested by BDR---give NA rather
        ## than 0 for non-estimable parameters
        if (fit$rank < nvars) coef[fit$pivot][seq.int(fit$rank+1, nvars)] <- NA
        xxnames <- xnames[fit$pivot]
        ## update by accurate calculation, including 0-weight cases.
        residuals <-  (y - mu)/mu.eta(eta)
##        residuals <- rep.int(NA, nobs)
##        residuals[good] <- z - (eta - offset)[good] # z does not have offset in.
        fit$qr <- as.matrix(fit$qr)
        nr <- min(sum(good), nvars)
        if (nr < nvars) {
            Rmat <- diag(nvars)
            Rmat[1L:nr, 1L:nvars] <- fit$qr[1L:nr, 1L:nvars]
        }
        else Rmat <- fit$qr[1L:nvars, 1L:nvars]
        Rmat <- as.matrix(Rmat)
        Rmat[row(Rmat) > col(Rmat)] <- 0
        names(coef) <- xnames
        colnames(fit$qr) <- xxnames
        dimnames(Rmat) <- list(xxnames, xxnames)
    }
    names(residuals) <- ynames
    names(mu) <- ynames
    names(eta) <- ynames
    # for compatibility with lm, which has a full-length weights vector
    wt <- rep.int(0, nobs)
    wt[good] <- w^2
    names(wt) <- ynames
    names(weights) <- ynames
    names(y) <- ynames
    if(!EMPTY)
        names(fit$effects) <-
            c(xxnames[seq_len(fit$rank)], rep.int("", sum(good) - fit$rank))
    ## calculate null deviance -- corrected in glm() if offset and intercept
    wtdmu <-
	if (intercept) sum(weights * y)/sum(weights) else linkinv(offset)
    nulldev <- sum(dev.resids(y, wtdmu, weights))
    ## calculate df
    n.ok <- nobs - sum(weights==0)
    nulldf <- n.ok - as.integer(intercept)
    rank <- if(EMPTY) 0 else fit$rank
    resdf  <- n.ok - rank
    ## calculate AIC
	if(exists('n')) {
		n <- get('n')
	} else {
		n <- NA
	}
    aic.model <-
	aic(y, n, mu, weights, dev) + 2*rank
	##     ^^ is only initialize()d for "binomial" [yuck!]
    list(coefficients = coef, residuals = residuals, fitted.values = mu,
	 effects = if(!EMPTY) fit$effects, R = if(!EMPTY) Rmat, rank = rank,
	 qr = if(!EMPTY) structure(fit[c("qr", "rank", "qraux", "pivot", "tol")], class="qr"),
         family = family,
	 linear.predictors = eta, deviance = dev, aic = aic.model,
	 null.deviance = nulldev, iter = iter, weights = wt,
	 prior.weights = weights, df.residual = resdf, df.null = nulldf,
	 y = y, converged = conv, boundary = boundary)
}