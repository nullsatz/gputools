// Lloyd's k-means, one step at a time. g holds x c' for a block of nr
// points against the k centres (nr x k, column-major); since
// |x - c|^2 = |x|^2 + |c|^2 - 2 x.c, and |x|^2 is the same for every
// centre, the nearest centre is the one with the least |c|^2 - 2 x.c.

// The nearest centre of each point in the block, as R's kmeans_Lloyd
// chooses it: the first of equals, and never a centre that is NaN (an
// empty cluster's). *changed is set if any label changes.
__global__ void gpuAssignNearest(const real * g, int nr, int k,
	const real * cnorm, int * labels, int * changed)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= nr) return;
	real best = REAL_INF;
	int inew = -1;
	for(int j = 0; j < k; j++) {
		real dd = cnorm[j] - real(2) * g[i + (size_t) j * nr];
		if(dd < best) {
			best = dd;
			inew = j;
		}
	}
	if(labels[i] != inew) {
		labels[i] = inew;
		*changed = 1;
	}
}

// a = the nr x k indicator matrix of the block's labels, so that a' x sums
// each cluster's points
__global__ void gpuOneHot(const int * labels, int nr, int k, real * a)
{
	size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
	if(e >= (size_t) nr * k) return;
	int i = e % nr, j = e / nr;
	a[e] = labels[i] == j ? real(1) : real(0);
}
