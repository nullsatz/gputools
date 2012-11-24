void SVRTrain(float * mexalpha, float * beta, float * y, float * x ,float C,
	float kernelwidth, float eps, int m, int n, float StoppingCrit,
	int * numSvs);

void SVMTrain(float *mexalpha,float* beta,float*y,float *x ,float C,
	float kernelwidth, int m, int n, float StoppingCrit,
	int * numSvs, int * numPosSvs);

void getSupportVectors(int isRegression, int m, int n, int numSVs,
	int numPosSVs, const float * x, const float * y, const float * alphas,
	float * svCoefficients, float * supportVectors);

void GPUPredictWrapper(int m, int n, int k, float kernelwidth,
	const float * Test, const float * Svs, float * alphas, float * prediction,
	float beta,float isregression);

double getAucEstimate(int n, double * classes, double * probs);


extern "C" {
	void R_SVRTrain(float * alpha, float * beta, float * y, float * x,
		float * C, float * kernelwidth, float * eps, int * m, int * n,
		float * StoppingCrit, int * numSvs);
	void R_SVMTrain(float * alpha, float * beta, float * y, float * x,
		float * C, float * kernelwidth, int * m, int * n, float * StoppingCrit,
		int * numSvs, int * numPosSvs);
	void R_produceSupportVectors(int * isRegression, int * m, int * n,
		int * numSVs, int * numPosSVs, const float * x, const float * y,
		const float * alphas, float * svCoefficients, float * supportVectors);
	void R_GPUPredictWrapper(int * m, int * n, int * k, float * kernelwidth,
		const float * Test, const float * Svs, float * alphas,
		float * prediction, float * beta, float * isregression);
	void RgetAucEstimate(int * n, double * classes, double * probs,
		double * outputAuc);
}
