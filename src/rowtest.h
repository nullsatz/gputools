#ifndef _ROWTEST_H_
#define _ROWTEST_H_

// Two-group t tests of each row of x (genes x n, column-major on the host,
// each row centred and scaled) under nb labellings of the samples: labels
// (n x nb) is 1 for group 1 and 0 for group 2, the same group sizes n1
// and n - n1 in every column, column 0 the observed labelling. t0 gets
// the observed t statistics; for the other columns, counts[g] how often
// |t| >= |t0| (1 - eps) and maxima[b - 1] the column's largest |t|.
template <typename T>
void rowTtests(int genes, int n, const T * x, int nb, const int * labels,
               int n1, bool pooled, double eps, double * t0, int * counts,
               double * maxima);

#endif /* _ROWTEST_H_ */
