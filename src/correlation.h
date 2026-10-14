typedef enum useObs {everything, pairwiseComplete} UseObs; 

template <typename T>
void pmcc(UseObs whichObs, const T * vectsa, size_t na,
	const T * vectsb, size_t nb, size_t dim, T * numPairs, 
          T * correlations, T * signifs);

void setDevice(int device);
void getDevice(int * device);

template <typename T>
void testSignif(const T * goodPairs, const T * coeffs, 
                size_t n, T * tscores);

template <typename T>
void hostSignif(const T * goodPairs, const T * coeffs, 
	size_t n, T * tscores);

