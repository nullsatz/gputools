#ifndef _QRDECOMP_H_
#define _QRDECOMP_H_

template <typename T>
void getQRDecompRR(int rows, int cols, double tol, T * dQR,
                   int * pivot, double * qrAux, int * rank);
template <typename T>
void getQRDecompLinpack(int rows, int cols, double tol, T * dQR,
                        int * pivot, double * qrAux, int * rank);


#endif /* _QRDECOMP_H_ */
