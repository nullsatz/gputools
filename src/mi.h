#ifndef _MI_H_
#define _MI_H_

// B-spline mutual information (bits) of every column of x (n x nx) with
// every column of y (n x ny; x itself when y is NULL), B-spline weights
// over `bins` bins of order `order`, the products in precision T.
// out[j + i*ny] is the information of y's column j with x's column i.
template <typename T>
void bSplineMutualInfo(int bins, int order, int n, int nx, const double * x,
                       int ny, const double * y, double * out);

#endif /* _MI_H_ */
