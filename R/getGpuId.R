getGpuId <- function()
{
	.Call(C_gpuGetDevice)
}
