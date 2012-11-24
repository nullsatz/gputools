% File man/gpuLm.defaultTol.Rd
% part of the gputools R package
% Distributed under GPL 2 or later

\name{gpuLm.defaultTol}
\title{Function to switch tolerance depending on precision}
\usage{
gpuLm.defaultTol(useSingle = TRUE)
}
\alias{gpuLm.defaultTol}
\description{
	This function was written by Mark Seligman at Rapid Biologics, 
	\code{http://rapidbiologics.com}

  The function \code{\link{gpuLm.fit}} calls this function 
  to determine a default tolerance. So gpuLm.defaultTol should
  \emph{not} need to be used directly.
}
\arguments{
  \item{useSingle}{logical. If TRUE, a tolerance will be returned appropriate
  	for single precision arithmetic. If FALSE, a tolerance will be returned
	appropriate for double precision arithmetic.}
}
\value{
	a floating point number representing a tolerance to be used by gpuLm.fit
}
\seealso{
  \code{\link{gpuLm.fit}} \code{\link{gpuLm}} 
}
