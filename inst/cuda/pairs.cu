// Selection from one tile of a correlation matrix: r holds the
// correlations of columns i0 .. i0 + ta - 1 of x with columns
// j0 .. j0 + tb - 1 of y, ta x tb column-major. The whole matrix is never
// held at once; the host steps through the tiles and keeps what these
// kernels pick out.

// how strong a correlation is: |r|, or r itself for positive ones only;
// NaN (a constant column) and excluded entries are never picked
__device__ bool strongEnough(real v, int absolute, real threshold)
{
	return (absolute ? fabs(v) : v) >= threshold;
}

// Every entry with strength >= threshold, appended to (outI, outJ, outR)
// at positions taken from *count. With lower set (x with itself), only
// entries with i < j are kept, so each pair appears once.
__global__ void gpuThresholdPairs(const real * r, int ta, int tb, int i0,
	int j0, int lower, int absolute, real threshold, int * count,
	int * outI, int * outJ, real * outR)
{
	size_t k = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
	if(k >= (size_t) ta * tb) return;
	int a = k % ta, b = k / ta, i = i0 + a, j = j0 + b;
	if(lower && i >= j) return;
	real v = r[k];
	if(!strongEnough(v, absolute, threshold)) return;
	int slot = atomicAdd(count, 1);
	outI[slot] = i;
	outJ[slot] = j;
	outR[slot] = v;
}

// sort key: larger is stronger; NaN, the diagonal (x with itself) and the
// padding sort last
__device__ real strength(real v, int absolute)
{
	return isnan(v) ? -REAL_INF : (absolute ? fabs(v) : v);
}

// For each column b of the tile, the k strongest entries in it, strongest
// first: their x columns in outI (-1 where the column has fewer than k
// usable entries) and values in outR, k per column. One block per column:
// a bitonic sort, descending, of (strength, position) in keys/pos, each
// column padded to n2, a power of two. With lower set (x with itself), the
// diagonal entry i == j is left out.
__global__ void gpuTopPerColumn(const real * r, int ta, int n2, int i0,
	int j0, int lower, int absolute, int k, real * keys, int * pos,
	int * outI, real * outR)
{
	size_t b = blockIdx.x;
	const real * rc = r + b * ta;
	real * kc = keys + b * n2;
	int * pc = pos + b * n2;
	int j = j0 + (int) b;

	for(int t = threadIdx.x; t < n2; t += blockDim.x) {
		bool usable = t < ta && !(lower && i0 + t == j);
		kc[t] = usable ? strength(rc[t], absolute) : -REAL_INF;
		pc[t] = usable && !isnan(rc[t]) ? t : -1;
	}
	__syncthreads();

	// descending by key; ties broken by position so the order is the same
	// on every run
	for(int size = 2; size <= n2; size <<= 1) {
		for(int stride = size >> 1; stride > 0; stride >>= 1) {
			for(int t = threadIdx.x; t < n2; t += blockDim.x) {
				int partner = t ^ stride;
				if(partner > t) {
					bool descending = (t & size) == 0;
					real ka = kc[t], kb = kc[partner];
					int pa = pc[t], pb = pc[partner];
					// does b belong before a?
					bool bFirst = kb > ka ||
						(kb == ka && (unsigned) pb < (unsigned) pa);
					if(descending ? bFirst : !bFirst) {
						kc[t] = kb; kc[partner] = ka;
						pc[t] = pb; pc[partner] = pa;
					}
				}
			}
			__syncthreads();
		}
	}

	for(int t = threadIdx.x; t < k; t += blockDim.x) {
		int p = t < n2 ? pc[t] : -1;
		outI[b * k + t] = p < 0 ? -1 : i0 + p;
		outR[b * k + t] = p < 0 ? REAL_NAN : rc[p];
	}
}
