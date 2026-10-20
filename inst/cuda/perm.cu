// Permutation tests of correlations: the columns of y, standardised, have
// their rows permuted, which leaves them standardised, so each
// permutation's correlations are one matrix product.

// out = z with its rows permuted: out[i, j] = z[perm[i], j] (n x q)
__global__ void gpuPermuteRows(const real * z, int n, int q, const int * perm,
	real * out)
{
	size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
	if(e >= (size_t) n * q) return;
	size_t i = e % n, j = e / n;
	out[e] = z[perm[i] + j * n];
}

// One permutation's correlations rb (p x q) against the observed r0:
// counts[e] += 1 where |rb| >= |r0| (1 - eps), a relative allowance for
// rounding in the comparison. Entries that are no test (NaN,
// or the diagonal of x with itself) are set to 0 in rb, so that its
// largest |value| is the permutation's maximum statistic.
__global__ void gpuCountExceed(real * rb, const real * r0, int p, int q,
	int self, real eps, int * counts)
{
	size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
	if(e >= (size_t) p * q) return;
	real v = rb[e], o = r0[e];
	if(isnan(v) || isnan(o) || (self && e % p == e / p)) {
		rb[e] = real(0);
		return;
	}
	if(fabs(v) >= fabs(o) * (real(1) - eps)) counts[e]++;
}
