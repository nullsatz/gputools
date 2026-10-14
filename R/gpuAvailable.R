gpuAvailable <- function()
{
	isTRUE(tryCatch(is.integer(getGpuId()), error = function(e) FALSE))
}
