// Two-group t tests of every row of x (genes x samples), under each of
// several labellings of the samples at once. s1 and q1 (genes x nb) are
// the sums and sums of squares of each row over the samples labelled
// group 1 in each labelling, s and q the row's totals; group 2 is the
// rest. Each row has been centred and scaled (which changes no t), so the
// sums of squares do not cancel.

// t (genes x nb): Welch's t, or Student's with the pooled variance if
// pooled is set; NaN where a variance is zero
__global__ void gpuRowT(const real * s1, const real * q1, const real * s,
	const real * q, int genes, int nb, int n1, int n2, int pooled,
	real * t)
{
	size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
	if(e >= (size_t) genes * nb) return;
	int g = e % genes;
	real a1 = s1[e], b1 = q1[e], a2 = s[g] - a1, b2 = q[g] - b1;
	real m1 = a1 / n1, m2 = a2 / n2;
	real ss1 = fmax(b1 - a1 * m1, real(0)), ss2 = fmax(b2 - a2 * m2, real(0));
	real se2;
	if(pooled) {
		real v = (ss1 + ss2) / (n1 + n2 - 2);
		se2 = v * (real(1) / n1 + real(1) / n2);
	} else {
		se2 = ss1 / (n1 - 1) / n1 + ss2 / (n2 - 1) / n2;
	}
	t[e] = se2 > real(0) ? (m1 - m2) / sqrt(se2) : REAL_NAN;
}

// For a block of c labellings, global columns first .. first + c - 1 of
// which column 0 is the observed one (skipped): counts[g] += 1 where
// |t_b| >= |t0| (1 - eps), and maxima[b - 1] = the largest |t_b| over the
// genes (NaN counting as 0). One block of 256 threads per column.
__global__ void gpuRowTCount(const real * t, const real * t0, int genes,
	int first, real eps, int * counts, double * maxima)
{
	__shared__ double best[256];
	int b = first + blockIdx.x;
	if(b == 0) return;
	const real * tb = t + (size_t) blockIdx.x * genes;
	double m = 0;
	for(int g = threadIdx.x; g < genes; g += blockDim.x) {
		real v = tb[g], o = t0[g];
		if(isnan(v)) continue;
		m = fmax(m, (double) fabs(v));
		if(!isnan(o) && fabs(v) >= fabs(o) * (real(1) - eps))
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
	if(threadIdx.x == 0) maxima[b - 1] = best[0];
}
