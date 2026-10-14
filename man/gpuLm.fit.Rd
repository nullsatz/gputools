% File man/gpuLm.fit.Rd
% part of the gputools R package
% mostly copied from
% File src/library/stats/man/lmfit.Rd
% Part of the R package, http://www.R-project.org
% Copyright 1995-2009 R Core Development Team
% Distributed under GPL 2 or later

\name{gpuLm.fit}
\title{Fitter functions for gpu enabled linear models}
\usage{
gpuLm.fit(x, y, w = NULL, offset = NULL, method = "qr",
	useSingle = getOption("gputools.precision", "single") == "single",
	tol = gpuLm.defaultTol(useSingle), singular.ok = TRUE, ...)
}
\alias{gpuLm.fit}
\description{
	The C code called by this function was written by
	Mark Seligman at Rapid Biologics, \code{http://rapidbiologics.com}

  The function \code{\link{gpuLm}} calls this function 
  to fit linear models. So gpuLm.fit should \emph{not} need to be used
  directly.
}
\arguments{
  \item{x}{design matrix of dimension \code{n * p}.}
  \item{y}{vector of observations of length \code{n}, or a matrix with
    \code{n} rows.}
  \item{w}{vector of weights (length \code{n}) to be used in the fitting
    process for the \code{wfit} functions.  Weighted least squares is
    used with weights \code{w}, i.e., \code{sum(w * e^2)} is minimized.}
  \item{offset}{numeric of length \code{n}).  This can be used to
    specify an \emph{a priori} known component to be included in the
    linear predictor during fitting.}

  \item{method}{currently, only \code{method="qr"} is supported.}

  \item{useSingle}{logical: \code{TRUE} computes in single precision on the GPU,
		\code{FALSE} in double. The default follows
		\code{options(gputools.precision)}: single unless that is \code{"double"}.
		It also sets the default tolerance (see \code{\link{gpuLm.defaultTol}}).}
  
  % Avoid Matrix's grab of qr.
  \item{tol}{tolerance for the \code{\link[base]{qr}} decomposition.  Default
    is 1e-7.}

  \item{singular.ok}{logical. If \code{FALSE}, a singular model is an
    error.}
  
  \item{\dots}{currently disregarded.}
}
\value{
%% S(-PLUS) returns an object of class "lm"
%% such that print.lm, summary,... work; but that'd need more changes for R.
  a list with components
  \item{coefficients}{\code{p} vector}
  \item{residuals}{\code{n} vector or matrix}
  \item{fitted.values}{\code{n} vector or matrix}
  \item{effects}{(not null fits)\code{n} vector of orthogonal single-df
    effects.  The first \code{rank} of them correspond to non-aliased
    coefficients, and are named accordingly.}
  \item{weights}{\code{n} vector --- \emph{only} for the \code{*wfit*}
    functions.}
  \item{rank}{integer, giving the rank}
  \item{df.residual}{degrees of freedom of residuals}
  \item{qr}{(not null fits) the QR decomposition, see \code{\link[base]{qr}}.}
}
\seealso{
  \code{\link{gpuLm}} which should usually be used for linear least squares
  regression
}
\examples{
\dontshow{if (gpuAvailable()) withAutoprint(\{ # examplesIf}
require(utils)
set.seed(129)
n <- 7 ; p <- 2
X <- matrix(rnorm(n * p), n,p) # no intercept!
y <- rnorm(n)
w <- rnorm(n)^2

str(lmw <- gpuLm.fit(x=X, y=y, w=w))
\dontshow{\}) # examplesIf}
}
\keyword{regression}
\keyword{array}
