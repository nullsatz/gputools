\name{getGpuId}
\alias{getGpuId}
\title{Discover the Id of the current GPU device}

\description{
	Queries the CUDA driver for the GPU device currently assigned to this
	thread. This is the id of the device that will be used for computation.
	If you wish to use a different device, use the chooseGpu function.
}

\usage{
getGpuId()
}

\value{
	The function returns a single integer indicating the id of the
	GPU device currently selected to carry out computation according to the
	CUDA driver.
}

\examples{
getGpuId()
}
