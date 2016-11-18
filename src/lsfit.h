#ifndef _LSFIT_H_
#define _LSFIT_H_

void getCRE(float *dQR, int rows, int cols, int stride, int rank,
            double *qrAux, int yCols,
            float *coeffs, float *resids, float *effects);
void gpuLSFitF(float *X, int n, int p, float *Y, int nY,
               double tol, float *coeffs,
               float *resids, float *effects,
               int *rank, int *pivot, double * qrAux);
void gpuLSFitD(double *X, int n, int p, double *Y, int nY,
               double tol, double *coeffs,
               double *resids, double *effects,
               int *rank, int *pivot, double * qrAux);

#endif /* _LSFIT_H_ */
