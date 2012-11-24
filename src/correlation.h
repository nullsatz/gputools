typedef enum useObs {everything, pairwiseComplete} UseObs; 

void pmcc(UseObs whichObs, const float * vectsa, size_t na,
	const float * vectsb, size_t nb, size_t dim, float * numPairs, 
	float * correlations, float * signifs);

void setDevice(int device);
void getDevice(int * device);

void getData(const int * images, 
	const int * xcoords, const int * ycoords, const int * zcoords,
	const int * mins, const int * maxes,
	const float * evs, size_t numrows, size_t numimages, float * output);

size_t parseResults(const int * imageList1, size_t numImages1, 
	const int * imageList2, size_t numImages2,
	int structureid,
	double cutCorrelation, int cutPairs,
	const double * correlations, const double * signifs, const int * numPairs, 
	double * results);

int isSignificant(double signif, int df);
void testSignif(const float * goodPairs, const float * coeffs, 
	size_t n, float * tscores);
void hostSignif(const float * goodPairs, const float * coeffs, 
	size_t n, float * tscores);
size_t signifFilter(const double * data, size_t rows, double * results);
size_t gpuSignifFilter(const float * data, size_t rows, float * results);

void cublasPMCC(const float * sampsa, size_t numSampsA, const float * sampsb, 
	size_t numSampsB, size_t sampSize, float * res);

double hostKendall(const float * X, const float * Y, size_t n);
void permHostKendall(const float * a, size_t na, const float * b, size_t nb,
	size_t sampleSize, double * results);

void masterKendall(const float * x,  size_t nx, const float * y, size_t ny, 
	size_t sampleSize, double * results);
