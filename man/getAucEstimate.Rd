\name{getAucEstimate}
\alias{getAucEstimate}
\title{Estimate the AUC of the ROC}

\description{
	This function gives a quick estimate of the area under the curve (AUC)
	of the receiver operating characteristic (ROC).  It is a quick way to
	estimate the quality of a binary classifier. The
	algorithm is based on a paper by David Hand and Robert Till
	(see references).
}

\usage{
getAucEstimate(classes, scores)
}

\arguments{
	\item{classes}{
		a vector of floating point numbers.  Each entry i
		corresponds to the real class of a point and should be either 0 or 1. 
		The negative class is represented by 0 and the positive class by 1.
		These entries correspond both in number and order to the same points
		associated with the scores vector.
	}
	\item{scores}{
		a vector of floating point numbers.  Each entry i corresponds to the
		probability that a point is in the positive class of a
		binary classification.  This will be the output of, for example, a
		binary classifier based on logistic regression.  These entries should
		correspond both in number and order to the same points associated with
		the classes vector.
	}
}

\value{
a single floating point number of double precision.  This number represents an 
estimate of the auc score for the algorithm responsible for the scores vector.
The estimation is according to the method of David Hand and Robert Till (see
references).
}

\references{
	Hand, David J. and Till, Robert J. (2001). A simple generalisation of
	the area under the ROC curve for multiple class classification problems.
	\emph{Machine Learning}. 45, 171--186.
}

\examples{
# generate some fake data
classes <- round(runif(20, min = 0, max = 1))

# fake probability that point i is in the positive class
scores <- runif(20, min = 0, max = 1)

b <- getAucEstimate(classes, scores)
print(b)
}
