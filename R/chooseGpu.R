chooseGpu <- function(deviceId = 0) 
{
	deviceId <- as.integer(deviceId)
	.Call(C_gpuSetDevice, deviceId)
	invisible(deviceId)
}
