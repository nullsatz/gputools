gpuGranger <- function(x, y=NULL, lag)
{
	x <- as.matrix(x)
	rows <- nrow(x)
	colsx <- ncol(x)

	if(rows - lag <= 2 * lag + 1) {
		stop("time sequence too short for lag: use longer sequences or smaller lag")
	}

	if(!is.null(y)) {
		y <- as.matrix(y)
	}

	lag <- as.integer(lag)

	if(is.null(y)) {
		colsy <- colsx
		cRetVal <- .C("rgpuGranger",
			as.integer(rows), as.integer(colsx), as.single(x), lag, 
			fStats = single(colsx*colsy), pValues = single(colsx*colsy),
			PACKAGE='gputools')
	} else {
		colsy <- ncol(y)
		cRetVal <- .C("rgpuGrangerXY",
			as.integer(rows), as.integer(colsx), as.single(x), 
			as.integer(colsy), as.single(y), lag, 
			fStats = single(colsx*colsy), pValues = single(colsx*colsy),
			PACKAGE='gputools')
	}
	fStats <- matrix(cRetVal$fStats, colsx, colsy)
	pValues <- matrix(cRetVal$pValues, colsx, colsy)
	return(list(fStatistics = fStats, pValues = pValues))
}
