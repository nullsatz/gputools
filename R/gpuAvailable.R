gpuAvailable <- function()
{
	reason <- tryCatch({ getGpuId(); NULL },
		error = function(e) conditionMessage(e))
	if (is.null(reason)) return(TRUE)
	structure(FALSE, reason = reason)
}
