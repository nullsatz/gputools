gpuCor <- function(x, y = NULL, use = "everything", method = "pearson") {
	x <- as.matrix(x)
	nx <- ncol(x)
	size <- nrow(x)

	if(is.null(y)) {
		y <- x
	} else {
		y <- as.matrix(y)
	}
	ny <- ncol(y)

	n <- nx * ny

    methods <- c("pearson", "kendall")
	method <- pmatch(method, methods, -1)
	if(is.na(method)) {
		stop("invalid correlation method")
	}
	if(method == -1) {
		stop("ambiguous correlation method")
	}

    uses <- c("everything", "pairwise.complete.obs")
	use <- pmatch(use, uses, -1)
	if(is.na(use)) {
		stop("invalid correlation method")
	}
	if(use == -1) {
		stop("ambiguous correlation method")
	}

	if(methods[method] == "pearson") {
		answer <- .C("rpmcc", NAOK=TRUE,
			as.integer(use - 1), as.single(x), as.integer(nx),
			as.single(y), as.integer(ny), as.integer(size),
			pairs = single(n), corr = single(n), ts = single(n),
			PACKAGE='gputools')

		pairs <- t(matrix(answer$pairs, ny, nx))
		corr <- t(matrix(answer$corr, ny, nx))
		ts <- t(matrix(answer$ts, ny, nx))

		return(list(coefficients = corr, ts = ts, pairs = pairs))

	} else if(methods[method] == "kendall") {

		if(uses[use] != "everything") {
			warning("NA handling for Kendall's is not yet supported. Defaulting to using everything. Sorry for any inconvenience.")
		}

		a <- .C("RgpuKendall",
			as.single(x), nx, as.single(y), ny, 
			size, result = double(nx*ny), PACKAGE = "gputools")

		pairs <- matrix(size, nx, ny)
		return(list(coefficients = matrix(a$result, nx, ny), pairs = pairs))
	} else {
		stop("This correlation method is not yet supported. Sorry for any inconvenience.")
	}
}

gpuTtest <- function(goodPairs, coeffs) {
	goodPairs <- as.single(goodPairs)
	coeffs <- as.single(coeffs)

	n <- as.integer(length(goodPairs))

	.C("rtestT", NAOK = TRUE,
		goodPairs, coeffs, n,
		results = single(n),
		PACKAGE = 'gputools')$results
}

#gpuSignifFilter <- function(olddata) {
#	rows <- as.integer(ncol(olddata))
#	olddata <- as.single(olddata)
#	newdata <- .C("gSignifFilter", NAOK = TRUE, PACKAGE = "gputools",
#		olddata, numRows = rows,
#		results = single(rows*6))
#	numRows <- newdata$numRows
#	results <- newdata$results[1:(numRows*6)]
#	if(numRows != 0) {
#		dim(results) <- c(6, numRows)
#		results <- t(results)
#	} else {
#		results <- NULL
#	}
#	results
#}
#
#pickGpu <- function(device = 0) {
#	device <- as.integer(device)
#	.C("rsetDevice", PACKAGE = "gputools", device)
#}
#
#getGpu <- function() {
#	.C("rgetDevice", PACKAGE = "gputools", device = integer(1))
#	device
#}
#
#formatPmccInput <- function(queryList, numImages, mins, maxes) {
#	images  <- as.integer(queryList$IMAGESERIESID)
#	xcoords <- as.integer(queryList$X)
#	ycoords <- as.integer(queryList$Y)
#	zcoords <- as.integer(queryList$Z)
#	mins <- as.integer(mins)
#	maxes <- as.integer(maxes)
#
#	xmax <- maxes[1]
#	xmin <- mins[1]
#	nx <- 1+abs(xmax - xmin)
#
#	ymax <- maxes[2]
#	ymin <- mins[2]
#	ny <- 1+abs(ymax - ymin)
#
#	zmax <- maxes[3]
#	zmin <- mins[3]
#	nz <- 1+abs(zmax - zmin)
#
#	size <- nx*ny*nz
#	mins <- as.integer(c(xmin, ymin, zmin))
#	maxes <- as.integer(c(xmax, ymax, zmax))
#
#	evs <- as.single(queryList$EV)
#	numrows <- as.integer(length(images))
#	numimages <- as.integer(numImages)
#
#	output <- single(numimages*size)
#	output[1:length(output)] <- NA
#	output <- .C("rformatInput", NAOK = TRUE, PACKAGE="gputools",
#		images, xcoords, ycoords, 
#		zcoords, mins, maxes, evs, numrows, numimages, output = output)$output
#	output <- matrix(output, size, numimages)
#}
#
# formatPmccOutput <- function(imagesA, imagesB, correlations, ts, pairCounts, 
# 	structureid = 0, corrCut = -1.0, pairCut = 0) 
# {
# 	imagesA <- as.integer(imagesA)
# 	nA <- as.integer(length(imagesA))
# 
# 	imagesB <- as.integer(imagesB)
# 	nB <- as.integer(length(imagesB))
# 
# 	pairCounts <- as.integer(pairCounts)
# 
# 	structureid <- as.integer(structureid)
# 	pairCut <- as.integer(pairCut)
# 
# 	corrCut <- as.double(corrCut)
# 	correlations <- as.double(correlations)
# 	ts <- as.double(ts)
# 
# 	output <- .C("rformatOutput", NAOK = TRUE, PACKAGE="gputools",
# 		imagesA, nA, imagesB, nB,
# 		structureid, corrCut, pairCut,
# 		correlations, ts, pairCounts, 
# 		results = double(nA*nB*6), numRows = integer(1))
# 
# 	numRows <- output$numRows
# 	results <- output$results[1:(numRows*6)]
# 	if(numRows != 0) {
# 		dim(results) <- c(6, numRows)
# 		results <- t(results)
# 	} else {
# 		results <- NULL
# 	}
# 	results
# }
#
#hostSignifFilter <- function(olddata) {
#	rows <- as.integer(ncol(olddata))
#	olddata <- as.double(olddata)
#	newdata <- .C("rSignifFilter", NAOK = TRUE, PACKAGE="gputools", 
#		olddata, numRows = rows,
#		results = double(rows*6))
#	numRows <- newdata$numRows
#	results <- newdata$results[1:(numRows*6)]
#	if(numRows != 0) {
#		dim(results) <- c(6, numRows)
#		results <- t(results)
#	} else {
#		results <- NULL
#	}
#	results
#}
#
#hostTtest <- function(goodPairs, coeffs) {
#	n <- as.integer(length(goodPairs))
#	goodPairs <- as.single(goodPairs)
#	coeffs <- as.single(coeffs)
#
#	.C("rhostT", NAOK = TRUE, PACKAGE = "gputools", goodPairs, coeffs, n,
#		results = single(n))$results
#}
#
#hostKendall <- function(samplesA, samplesB) {
#	na <- as.integer(ncol(samplesA))
#	nb <- as.integer(ncol(samplesB))
#	numSamples <- as.integer(nrow(samplesA))
#	a <- .C("RpermHostKendall", PACKAGE = "gputools",
#		as.single(samplesA), na, as.single(samplesB), 
#		nb, numSamples, result = double(na*nb))
#	a$result
#	matrix(a$result, na, nb)
#}
