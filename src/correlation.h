typedef enum useObs {everything, pairwiseComplete} UseObs; 

void pmcc(UseObs whichObs, const float * vectsa, size_t na,
	const float * vectsb, size_t nb, size_t dim, float * numPairs, 
          float * correlations, float * signifs);

void setDevice(int device);
void getDevice(int * device);



void testSignif(const float * goodPairs, const float * coeffs, 
                size_t n, float * tscores);

void hostSignif(const float * goodPairs, const float * coeffs, 
	size_t n, float * tscores);



void masterKendall(const float * x,  size_t nx, const float * y, size_t ny, 
                   size_t sampleSize, double * results);
