# The precision argument every GPU function takes. Single precision is the
# default: it is what the package has always computed in, and consumer GPUs
# run it many times faster than double.
.precision <- function(precision)
{
	if (!is.character(precision) || length(precision) != 1L ||
	    !(precision %in% c("single", "double"))) {
		stop("precision must be \"single\" or \"double\"")
	}
	precision
}
