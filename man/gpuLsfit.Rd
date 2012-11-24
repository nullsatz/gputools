% File man/gpuLsfit.Rd
% Part of the gputools package
% mostly copied from
% File src/library/stats/man/lsfit.Rd
% Part of the R package, http://www.R-project.org
% Copyright 1995-2007 R Core Development Team
% Distributed under GPL 2 or later

\name{gpuLsfit}
\alias{gpuLsfit}

\title{Least squares fit using GPU--enabled QR decomposition}

\usage{
gpuLsfit(x, y, wt=NULL, intercept=TRUE, useSingle = TRUE, tolerance=gpuLm.defaultTol(useSingle), yname=NULL)
}

\description{
The least squares estimate of \bold{\eqn{\beta}{b}} in the model
\deqn{\bold{Y} = \bold{X \beta} + \bold{\epsilon}}{y = X b + e}
is found.

Most of this documentation is copied from R's documentation for
\code{lsfit}. The function \code{gpuLsfit} performs a least--squares fit
using a GPU enabled QR decomposition.

Note:  The QR decomposition employed by \code{gpuLm} is optimized for speed
and uses minimal pivoting.  If more precise pivoting is desired, then
either the function \code{gpuQR} or, better still, \code{svd} should be used.
}

\arguments{
\item{x}{a matrix whose rows correspond to cases and whose columns
  correspond to variables.}
\item{y}{the responses, possibly a matrix if you want to fit multiple
  left hand sides.}
\item{wt}{an optional vector of weights for performing weighted least squares.}
\item{intercept}{whether or not an intercept term should be used.}
\item{useSingle}{whether to use single precision arithmetic on the gpu. Only the 'TRUE' option is implemented so far.}
\item{tolerance}{the tolerance to be used in the matrix decomposition.
  This defaults to 1e-04 for single--precision GPU computation.}
\item{yname}{names to be used for the response variables.}
}

\details{
If weights are specified then a weighted least squares is performed
with the weight given to the \emph{j}th case specified by the \emph{j}th
entry in \code{wt}.

If any observation has a missing value in any field, that observation
is removed before the analysis is carried out.
This can be quite inefficient if there is a lot of missing data.

The implementation is via a modification of the LINPACK subroutines
which allow for multiple left-hand sides.
}

\value{
A list with the following named components:
\item{coef}{the least squares estimates of the coefficients in
  the model (\bold{\eqn{\beta}{b}} as stated above).}
\item{residuals}{residuals from the fit.}
\item{intercept}{indicates whether an intercept was fitted.}
\item{qr}{the QR decomposition of the design matrix.}
}

\author{
This function was adapted for Nvidia's CUDA--supporting GPGPUs 
by Mark Seligman at Rapid Biologics LLC.  
	\code{http://www.rapidbiologics.com}
  }

\references{
  Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988)
  \emph{The New S Language}. Wadsworth & Brooks/Cole.
}
\seealso{
\code{\link{lsfit}}, \code{\link{lm}}, 
\code{\link{ls.print}}, \code{\link{ls.diag}}
}
\examples{
\dontshow{utils::example("lm", echo = FALSE)}
##-- Using the same data as the lm(.) example:
lsD9 <- gpuLsfit(x = unclass(gl(2,10)), y = weight)
ls.print(lsD9)
}
\keyword{regression}
