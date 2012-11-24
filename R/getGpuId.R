getGpuId <- function()
{
	deviceId <- .C("rgetDevice", deviceId = integer(1),
		PACKAGE='gputools')$deviceId
	return(deviceId)
}
