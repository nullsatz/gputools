#ifndef _GRANGER_H_
#define _GRANGER_H_

// Granger causality F tests of every cause (column of x) on every effect
// (column of y; x itself when y is NULL), lag past values each, in double.
// fStats and pValues are colsx x colsy, column-major, [i + j*colsx] the
// test of cause i on effect j; NaN where there is no test (a series on
// itself, a constant series) or it cannot be resolved in float64, and
// unresolved counts the latter.
void granger(int rows, int colsx, const double * x, int colsy,
             const double * y, int lag, double * fStats, double * pValues,
             int * unresolved);

#endif /* _GRANGER_H_ */
