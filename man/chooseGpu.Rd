\name{chooseGpu}
\alias{chooseGpu}
\title{Choose which GPU device to use}

\description{
	Selects the GPU device to use for computation. This is only useful on a
	machine equipped with multiple GPU devices. The numbering starts at 0
	and is assigned by the CUDA capable driver.

	Choosing a device can only be done before any other GPU operation and
	only once per thread.
}

\usage{
chooseGpu(deviceId = 0) 
}

\arguments{
	\item{deviceId}{an integer >= 0 designating the GPU to use for computation.}
}

\value{
	chooseGpu should print out an integer specifying the device id chosen or 
	an error message.
}
