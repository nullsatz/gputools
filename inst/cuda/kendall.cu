#define NUMTHREADS 16

// Rank correlations between column bx of a and column by of b, over the
// rows where both are present (pairwise complete observations, as cor()
// counts them with use = "pairwise.complete.obs"); NaN marks a missing
// value. One block per pair of columns; counts are kept as integers and
// rank sums as doubles, so neither loses anything to single precision.

// Kendall's tau-b, as cor() computes it:
//
//   (C - D) / sqrt((n0 - Tx) (n0 - Ty)),  n0 = m (m - 1) / 2
//
// over the m complete rows, where C and D count the concordant and
// discordant pairs and Tx, Ty the pairs tied in x and in y (a pair tied in
// both counts in both, and is neither concordant nor discordant).
__global__ void gpuKendall(const real * a, size_t na,
	const real * b, size_t nb, size_t sampleSize, double * results,
	double * counts)
{
	typedef unsigned long long count;
	const int nthreads = NUMTHREADS * NUMTHREADS;
	size_t
		tx = threadIdx.x, ty = threadIdx.y,
		bx = blockIdx.x, by = blockIdx.y,
		rowa = bx * sampleSize, rowb = by * sampleSize,
		t = tx * NUMTHREADS + ty;
	count c = 0, d = 0, tiesA = 0, tiesB = 0, complete = 0;

	__shared__ count sums[5 * nthreads];

	for(size_t i = tx; i < sampleSize; i += NUMTHREADS) {
		real ai = a[rowa+i], bi = b[rowb+i];
		if(isnan(ai) || isnan(bi))
			continue;
		if(ty == 0)
			++complete;
		for(size_t j = i+1+ty; j < sampleSize; j += NUMTHREADS) {
			real aj = a[rowa+j], bj = b[rowb+j];
			if(isnan(aj) || isnan(bj))
				continue;
			real da = aj - ai, db = bj - bi;
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
	sums[t + 4 * nthreads] = complete;
	__syncthreads();

	for(int stride = nthreads >> 1; stride > 0; stride >>= 1) {
		if(t < stride)
			for(int k = 0; k < 5; ++k)
				sums[t + k * nthreads] += sums[t + stride + k * nthreads];
		__syncthreads();
	}

	if(t == 0) {
		double
			m = (double) sums[4 * nthreads],
			n0 = 0.5 * m * (m - 1.0),
			numer = (double) sums[0] - (double) sums[nthreads],
			denom = sqrt((n0 - (double) sums[2 * nthreads]) *
			             (n0 - (double) sums[3 * nthreads]));
		results[by*na+bx] = numer / denom;
		counts[by*na+bx] = m;
	}
}

// Spearman's rho over the complete rows of a pair, for data with missing
// values: each value is ranked among that pair's complete rows only (ties
// get the average rank), as cor(method = "spearman",
// use = "pairwise.complete.obs") ranks them, and rho is the Pearson
// correlation of the ranks. Data without missing values is ranked once per
// column on the host instead.
__global__ void gpuSpearmanPairwise(const real * a, size_t na,
	const real * b, size_t nb, size_t sampleSize, double * results,
	double * counts)
{
	const int nthreads = NUMTHREADS * NUMTHREADS;
	size_t
		tx = threadIdx.x, ty = threadIdx.y,
		bx = blockIdx.x, by = blockIdx.y,
		rowa = bx * sampleSize, rowb = by * sampleSize,
		t = tx * NUMTHREADS + ty;
	double m = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;

	__shared__ double sums[6 * nthreads];

	for(size_t i = t; i < sampleSize; i += nthreads) {
		real ai = a[rowa+i], bi = b[rowb+i];
		if(isnan(ai) || isnan(bi))
			continue;
		// rank = 1 + (values below) + (other equal values) / 2
		double rx = 1, ry = 1;
		for(size_t k = 0; k < sampleSize; ++k) {
			real ak = a[rowa+k], bk = b[rowb+k];
			if(k == i || isnan(ak) || isnan(bk))
				continue;
			rx += (ak < ai) + 0.5 * (ak == ai);
			ry += (bk < bi) + 0.5 * (bk == bi);
		}
		m += 1;
		sx += rx;
		sy += ry;
		sxx += rx * rx;
		syy += ry * ry;
		sxy += rx * ry;
	}
	sums[t] = m;
	sums[t + nthreads] = sx;
	sums[t + 2 * nthreads] = sy;
	sums[t + 3 * nthreads] = sxx;
	sums[t + 4 * nthreads] = syy;
	sums[t + 5 * nthreads] = sxy;
	__syncthreads();

	for(int stride = nthreads >> 1; stride > 0; stride >>= 1) {
		if(t < stride)
			for(int k = 0; k < 6; ++k)
				sums[t + k * nthreads] += sums[t + stride + k * nthreads];
		__syncthreads();
	}

	if(t == 0) {
		double
			n = sums[0],
			cxy = n * sums[5 * nthreads] - sums[nthreads] * sums[2 * nthreads],
			cxx = n * sums[3 * nthreads] - sums[nthreads] * sums[nthreads],
			cyy = n * sums[4 * nthreads] - sums[2 * nthreads] * sums[2 * nthreads];
		results[by*na+bx] = cxy / sqrt(cxx * cyy);
		counts[by*na+bx] = n;
	}
}
