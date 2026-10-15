#ifndef _QRDECOMP_H_
#define _QRDECOMP_H_

template <typename T>
void getQRDecompRR(int rows, int cols, double tol, T * dQR,
                   int * pivot, double * qrAux, int * rank);
template <typename T>
void getQRDecompLinpack(int rows, int cols, double tol, T * dQR,
                        int * pivot, double * qrAux, int * rank);

template <typename T>
void getInverseFromQR(int rows, int cols, const T * dQ, const T * dR, 
	T * dInverse);
template <typename T>
void solveFromQR(int rows, int cols, const T * matQ, const T * matR,
	const T * vectY, T * vectB);

#endif /* _QRDECOMP_H_ */
