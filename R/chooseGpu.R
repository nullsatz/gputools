chooseGpu <- function(deviceId = 0) 
{
	deviceId <- as.integer(deviceId)
	.C("rsetDevice", deviceId, PACKAGE='gputools')
}
