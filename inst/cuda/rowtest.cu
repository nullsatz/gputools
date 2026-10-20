// Two-group t tests of every row of x (genes x samples), under each of
// several labellings of the samples at once. s1 and q1 (genes x nb) are
// the sums and sums of squares of each row over the samples labelled
// group 1 in each labelling, s and q the row's totals; group 2 is the
// rest. Each row has been centred and scaled (which changes no t), so the
// sums of squares do not cancel.

// t (genes x nb): Welch's t, or Student's with the pooled variance if
// pooled is set. A within-group sum of squares within rounding of zero
// (16 eps of the sum of squares it is taken from) is zero: a row constant
// within the group would otherwise get a large finite t where the true t
// is infinite. A zero standard error with the means apart gives +/-Inf
// (a labelling that separates the groups completely); with the means
// equal, or a constant row (NaN after scaling), there is no test: NaN.
__global__ void gpuRowT(const real * s1, const real * q1, const real * s,
	const real * q, int genes, int nb, int n1, int n2, int pooled,
	real * t)
{
	size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
	if(e >= (size_t) genes * nb) return;
	int g = e % genes;
	real a1 = s1[e], b1 = q1[e], a2 = s[g] - a1, b2 = q[g] - b1;
	real m1 = a1 / n1, m2 = a2 / n2;
	const real tiny = real(16) * (sizeof(real) == 4 ? real(1.1920929e-7)
	                                                : real(2.220446049250313e-16));
	real ss1 = b1 - a1 * m1, ss2 = b2 - a2 * m2;
	if(ss1 <= tiny * b1) ss1 = real(0);
	if(ss2 <= tiny * b2) ss2 = real(0);
	real se2;
	if(pooled) {
		real v = (ss1 + ss2) / (n1 + n2 - 2);
		se2 = v * (real(1) / n1 + real(1) / n2);
	} else {
		se2 = ss1 / (n1 - 1) / n1 + ss2 / (n2 - 1) / n2;
	}
	real diff = m1 - m2;
	if(se2 > real(0)) t[e] = diff / sqrt(se2);
	else if(diff > real(0)) t[e] = REAL_INF;
	else if(diff < real(0)) t[e] = -REAL_INF;
	else t[e] = REAL_NAN;
}

// For a block of labellings whose column 0 is the observed one, recomputed
// with the block (so a labelling equivalent to it is compared with the
// same arithmetic), and columns 1 .. c the permutations first .. first +
// c - 1: counts[g] += 1 where |t_b| >= |t0| (1 - eps), and maxima[first +
// b - 1] = the largest |t_b| over the rows that have a test (a row with
// no observed test, NaN, counts as 0 everywhere). One block of 256
// threads per permutation column.
__global__ void gpuRowTCount(const real * t, int genes, int first, real eps,
	int * counts, double * maxima)
{
	__shared__ double best[256];
	const real * t0 = t;
	const real * tb = t + (size_t) (blockIdx.x + 1) * genes;
	double m = 0;
	for(int g = threadIdx.x; g < genes; g += blockDim.x) {
		real v = tb[g], o = t0[g];
		if(isnan(o) || isnan(v)) continue;
		m = fmax(m, (double) fabs(v));
		if(fabs(v) >= fabs(o) * (real(1) - eps))
			atomicAdd(counts + g, 1);
	}
	best[threadIdx.x] = m;
	__syncthreads();
	for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
		if(threadIdx.x < stride)
			best[threadIdx.x] = fmax(best[threadIdx.x],
			                         best[threadIdx.x + stride]);
		__syncthreads();
	}
	if(threadIdx.x == 0) maxima[first + blockIdx.x] = best[0];
}
