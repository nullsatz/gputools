// Ranks each column of x (n rows, column-major) as
// rank(ties.method = "average", na.last = "keep") does: ties get the
// average of the ranks they span, NaN stays NaN. One block per column:
// a bitonic sort of the column's (value, row) pairs in keys/rows (each
// column padded to n2, a power of two), then each value's tied run is
// found by binary search in the sorted column.

// order classes: values first, then NaN, then the padding
__device__ int rankClass(real v, int row)
{
	return row < 0 ? 2 : (isnan(v) ? 1 : 0);
}

__device__ bool rankBefore(real a, int ra, real b, int rb)
{
	int ca = rankClass(a, ra), cb = rankClass(b, rb);
	if(ca != cb) return ca < cb;
	return ca == 0 && a < b;
}

__global__ void gpuRankColumns(const real * x, int n, int n2, real * keys,
	int * rows, real * ranks)
{
	size_t col = blockIdx.x;
	const real * xc = x + col * n;
	real * kc = keys + col * n2;
	int * rc = rows + col * n2;
	real * out = ranks + col * n;

	for(int i = threadIdx.x; i < n2; i += blockDim.x) {
		kc[i] = i < n ? xc[i] : real(0);
		rc[i] = i < n ? i : -1;
	}
	__syncthreads();

	for(int k = 2; k <= n2; k <<= 1) {
		for(int j = k >> 1; j > 0; j >>= 1) {
			for(int i = threadIdx.x; i < n2; i += blockDim.x) {
				int partner = i ^ j;
				if(partner > i) {
					bool ascending = (i & k) == 0;
					real a = kc[i], b = kc[partner];
					int ra = rc[i], rb = rc[partner];
					bool outOfOrder = ascending ? rankBefore(b, rb, a, ra)
					                            : rankBefore(a, ra, b, rb);
					if(outOfOrder) {
						kc[i] = b; kc[partner] = a;
						rc[i] = rb; rc[partner] = ra;
					}
				}
			}
			__syncthreads();
		}
	}

	// m values sort first; NaN rows follow
	__shared__ int m;
	if(threadIdx.x == 0) {
		int lo = 0, hi = n2;
		while(lo < hi) {
			int mid = (lo + hi) / 2;
			if(rankClass(kc[mid], rc[mid]) == 0) lo = mid + 1; else hi = mid;
		}
		m = lo;
	}
	__syncthreads();

	for(int p = threadIdx.x; p < n2; p += blockDim.x) {
		int row = rc[p];
		if(row < 0) continue;
		if(p >= m) {
			out[row] = REAL_NAN;
			continue;
		}
		real v = kc[p];
		int lo = 0, hi = m;                 // first position with value v
		while(lo < hi) {
			int mid = (lo + hi) / 2;
			if(kc[mid] < v) lo = mid + 1; else hi = mid;
		}
		int first = lo;
		lo = first; hi = m;                 // one past the last
		while(lo < hi) {
			int mid = (lo + hi) / 2;
			if(kc[mid] <= v) lo = mid + 1; else hi = mid;
		}
		// positions first..lo-1 hold ranks first+1..lo: their average
		out[row] = real(0.5) * (real(first + 1) + real(lo));
	}
}
