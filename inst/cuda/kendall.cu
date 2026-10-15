#define NUMTHREADS 16

// Kendall's tau-b of column bx of a with column by of b, as cor() computes
// it:
//
//   (C - D) / sqrt((n0 - Tx) (n0 - Ty))
//
// over the n0 = n (n - 1) / 2 pairs of observations, where C and D count
// the concordant and discordant pairs and Tx, Ty the pairs tied in x and
// in y (a pair tied in both counts in both, and is neither concordant nor
// discordant). The counts are integers so they stay exact however many
// pairs there are.
__global__ void gpuKendall(const real * a, size_t na,
	const real * b, size_t nb, size_t sampleSize, double * results)
{
	typedef unsigned long long count;
	const int nthreads = NUMTHREADS * NUMTHREADS;
	size_t
		tx = threadIdx.x, ty = threadIdx.y,
		bx = blockIdx.x, by = blockIdx.y,
		rowa = bx * sampleSize, rowb = by * sampleSize,
		t = tx * NUMTHREADS + ty;
	count c = 0, d = 0, tiesA = 0, tiesB = 0;

	__shared__ count sums[4 * nthreads];

	for(size_t i = tx; i < sampleSize; i += NUMTHREADS) {
		real ai = a[rowa+i], bi = b[rowb+i];
		for(size_t j = i+1+ty; j < sampleSize; j += NUMTHREADS) {
			real da = a[rowa+j] - ai, db = b[rowb+j] - bi;
			tiesA += (da == real(0));
			tiesB += (db == real(0));
			c += (da > real(0) && db > real(0)) || (da < real(0) && db < real(0));
			d += (da > real(0) && db < real(0)) || (da < real(0) && db > real(0));
		}
	}
	sums[t] = c;
	sums[t + nthreads] = d;
	sums[t + 2 * nthreads] = tiesA;
	sums[t + 3 * nthreads] = tiesB;
	__syncthreads();

	for(int stride = nthreads >> 1; stride > 0; stride >>= 1) {
		if(t < stride)
			for(int k = 0; k < 4; ++k)
				sums[t + k * nthreads] += sums[t + stride + k * nthreads];
		__syncthreads();
	}

	if(t == 0) {
		double
			n0 = 0.5 * (double) sampleSize * ((double) sampleSize - 1.0),
			numer = (double) sums[0] - (double) sums[nthreads],
			denom = sqrt((n0 - (double) sums[2 * nthreads]) *
			             (n0 - (double) sums[3 * nthreads]));
		results[by*na+bx] = numer / denom;
	}
}
