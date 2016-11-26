#define NUMTHREADS 16
#define THREADWORK 32

template<typename T>
__global__ void gpuKendall(const T * a, size_t na,
                           const T * b, size_t nb,
                           size_t sampleSize,
                           double * results) 
{
	size_t 
		i, j, tests, 
		tx = threadIdx.x, ty = threadIdx.y, 
		bx = blockIdx.x, by = blockIdx.y,
		rowa = bx * sampleSize, rowb = by * sampleSize;
	T 
		discordant, concordant = 0.0,
		numer, denom;

	__shared__ T threadSums[NUMTHREADS*NUMTHREADS];

	for(i = tx; i < sampleSize; i += NUMTHREADS) {
		for(j = i+1+ty; j < sampleSize; j += NUMTHREADS) {
			tests = ((a[rowa+j] >  a[rowa+i]) && (b[rowb+j] >  b[rowb+i]))
				+ ((a[rowa+j] <  a[rowa+i]) && (b[rowb+j] <  b[rowb+i])) 
				+ ((a[rowa+j] == a[rowa+i]) && (b[rowb+j] == b[rowb+i])); 
			concordant = concordant + (double) tests;
		}
	}
	threadSums[tx*NUMTHREADS+ty] = concordant;

	__syncthreads();
	for(i = NUMTHREADS >> 1; i > 0; i >>= 1) {
		if(ty < i)
			threadSums[tx*NUMTHREADS+ty] += threadSums[tx*NUMTHREADS+ty+i];
		__syncthreads();
	}
  for(i = NUMTHREADS >> 1; i > 0; i >>= 1) {
    if((tx < i) && (ty == 0))
      threadSums[tx*NUMTHREADS] += threadSums[(tx+i)*NUMTHREADS];
    __syncthreads();
  }

	if((tx == 0) && (ty == 0)) {
		concordant = threadSums[0];
		denom = (double) sampleSize;
		denom = (denom * (denom - 1.f)) / 2.f; discordant = denom - concordant;
		numer = concordant - discordant;
		results[by*na+bx] = ((double) numer) / ((double) denom);
	}
}
