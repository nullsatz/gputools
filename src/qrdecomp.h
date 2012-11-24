void qrdecompMGS(int rows, int cols, float * da, float * dq, float * dr, 
	int * pivots);
void getQRDecomp(int rows, int cols, float * dq, float * da, int * pivot);
void qrSolver(int rows, int cols, float * matX, float * vectY, float * vectB);
void getQRDecompRR(int rows, int cols, double tol, float * dQR,
	int * pivot, double * qrAux, int * rank);
void getQRDecompBlocked(int rows, int cols, double tol, float * dQR,
	int blockSize, int rowsUnblocked, int * pivot, double * qrAux, int * rank);

void getInverseFromQR(int rows, int cols, const float * dQ, const float * dR, 
	float * dInverse);
void solveFromQR(int rows, int cols, const float * matQ, const float * matR,
	const float * vectY, float * vectB);
