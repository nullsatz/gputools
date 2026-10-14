\name{gpuAvailable}
\alias{gpuAvailable}
\title{Is a CUDA Device Usable?}
\description{
  Reports whether gputools can use a CUDA device: the NVIDIA driver is
  installed and at least one GPU is visible. The GPU functions stop with
  an error when it is \code{FALSE}.
}
\usage{
gpuAvailable()
}
\value{\code{TRUE} or \code{FALSE}.}
\seealso{\code{\link{chooseGpu}}, \code{\link{getGpuId}}}
\examples{
if (gpuAvailable()) getGpuId()
}
\keyword{utilities}
