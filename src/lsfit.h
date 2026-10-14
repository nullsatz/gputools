#ifndef _LSFIT_H_
#define _LSFIT_H_

template <typename T>
void getCRE(T *dQR, int rows, int cols, int stride, int rank,
            double *qrAux, int yCols,
            T *coeffs, T *resids, T *effects);
template <typename T>
void leastSquaresFit(T *X, int n, int p, T *Y, int nY,
               double tol, T *coeffs,
               T *resids, T *effects,
               int *rank, int *pivot, double * qrAux);
#endif /* _LSFIT_H_ */
