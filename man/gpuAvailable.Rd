\name{gpuAvailable}
\alias{gpuAvailable}
\title{Is a CUDA Device Usable?}
\description{
  Reports whether gputools can use a CUDA device: the package was built
  with CUDA, the NVIDIA driver is installed and at least one GPU is
  visible. The GPU functions stop with an error when it is \code{FALSE}.
}
\details{
  A gputools installed where configure could not find the CUDA toolkit
  (for example a binary built on a server without CUDA) loads, but its
  GPU functions cannot run; it says so when attached. Install the NVIDIA
  CUDA toolkit (11.7 or later), make it findable -- set \env{CUDA_HOME} to
  its directory, or put \command{nvcc} on the \env{PATH} -- and reinstall
  gputools from source.
}
\usage{
gpuAvailable()
}
\value{\code{TRUE}, or \code{FALSE} with a \code{"reason"} attribute
  saying why no device can be used.}
\seealso{\code{\link{chooseGpu}}, \code{\link{getGpuId}}}
\examples{
if (gpuAvailable()) getGpuId()
}
\keyword{utilities}
