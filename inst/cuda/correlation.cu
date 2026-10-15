#define FALSE 0
#define TRUE !FALSE

#define NUMTHREADS 16
#define THREADWORK 32

__global__ void gpuMeans(const real * vectsA, size_t na, 
	const real * vectsB, size_t nb, size_t dim, 
	real * means, real * numPairs)
{
	size_t 
		offset, stride,
		bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;
	real a, b;

	__shared__ real 
		threadSumsA[NUMTHREADS], threadSumsB[NUMTHREADS],
		count[NUMTHREADS];

	if((bx >= na) || (by >= nb))
		return;

	threadSumsA[tx] = real(0);
	threadSumsB[tx] = real(0);
	count[tx] = real(0);

	for(offset = tx; offset < dim; offset += NUMTHREADS) {
		a = vectsA[bx * dim + offset];
		b = vectsB[by * dim + offset];
		if(!(isnan(a) || isnan(b))) {
			threadSumsA[tx] += a;
			threadSumsB[tx] += b;
			count[tx] += real(1);
		}
	}
	__syncthreads();
    
	for(stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
		if(tx < stride) {
			threadSumsA[tx] += threadSumsA[tx + stride];
			threadSumsB[tx] += threadSumsB[tx + stride];
			count[tx] += count[tx+stride];
		}
		__syncthreads();
	}
	if(tx == 0) {
		means[bx*nb*2+by*2] = threadSumsA[0] / count[0];
		means[bx*nb*2+by*2+1] = threadSumsB[0] / count[0];
		numPairs[bx*nb+by] = count[0];
	}
}

__global__ void gpuSD(const real * vectsA, size_t na,
	const real * vectsB, size_t nb, size_t dim, 
	const real * means, const real * numPairs, real * sds)
{
	size_t 
		offset, stride,
		tx = threadIdx.x, 
		bx = blockIdx.x, by = blockIdx.y;
	real 
		a, b,
		termA, termB;
	__shared__ real 
		meanA, meanB, n,
		threadSumsA[NUMTHREADS], threadSumsB[NUMTHREADS];

	if((bx >= na) || (by >= nb))
		return;

	if(tx == 0) {
		meanA = means[bx*nb*2+by*2];	
		meanB = means[bx*nb*2+by*2+1];	
		n = numPairs[bx*nb+by]; 
	}
	__syncthreads();

	threadSumsA[tx] = real(0);
	threadSumsB[tx] = real(0);
	for(offset = tx; offset < dim; offset += NUMTHREADS) {
		a = vectsA[bx * dim + offset];
		b = vectsB[by * dim + offset];
		if(!(isnan(a) || isnan(b))) {
			termA = a - meanA;
			termB = b - meanB;
			threadSumsA[tx] += termA * termA;
			threadSumsB[tx] += termB * termB;
		}
	}
	__syncthreads();

	for(stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
		if(tx < stride) {
			threadSumsA[tx] += threadSumsA[tx + stride];
			threadSumsB[tx] += threadSumsB[tx + stride];
		}
		__syncthreads();
	}
	if(tx == 0) {
		sds[bx*nb*2+by*2]   = sqrt(threadSumsA[0] / (n - real(1)));
		sds[bx*nb*2+by*2+1] = sqrt(threadSumsB[0] / (n - real(1)));
	}
}

__global__ void gpuPMCC(const real * vectsa, size_t na,
	const real * vectsb, size_t nb, size_t dim,
	const real * numPairs, const real * means, const real * sds,
	real * correlations) 
{
	size_t 
		offset, stride,
		x = blockIdx.x, y = blockIdx.y, 
		tx = threadIdx.x;
	real 
		a, b, n, scoreA, scoreB;
    __shared__ real 
		meanA, meanB, 
		sdA, sdB, 
		threadSums[NUMTHREADS];

	if((x >= na) || (y >= nb))
		return;

	if(tx == 0) {
		meanA = means[x*nb*2+y*2];
		meanB = means[x*nb*2+y*2+1];	
		sdA = sds[x*nb*2+y*2];
		sdB = sds[x*nb*2+y*2+1];	
		n = numPairs[x*nb+y]; 
	}
	__syncthreads();

	threadSums[tx] = real(0);
	for(offset = tx; offset < dim; offset += NUMTHREADS) {
		a = vectsa[x * dim + offset];
		b = vectsb[y * dim + offset];
		if(!(isnan(a) || isnan(b))) {
			scoreA = (a - meanA) / sdA; 
			scoreB = (b - meanB) / sdB;
			threadSums[tx] += scoreA * scoreB;
		}
	}
	__syncthreads();

	for(stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
		if(tx < stride) threadSums[tx] += threadSums[tx + stride];
		__syncthreads();
	}
	if(tx == 0) correlations[x*nb+y] = threadSums[0] / (n - real(1));
}

__global__ void gpuMeansNoTest(const real * vectsA, size_t na, 
	const real * vectsB, size_t nb, size_t dim, 
	real * means, real * numPairs)
{
	size_t 
		offset, stride,
		bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;
	real a, b;

	__shared__ real 
		threadSumsA[NUMTHREADS], threadSumsB[NUMTHREADS],
		count[NUMTHREADS];

	if((bx >= na) || (by >= nb))
		return;

	threadSumsA[tx] = real(0);
	threadSumsB[tx] = real(0);
	count[tx] = real(0);

	for(offset = tx; offset < dim; offset += NUMTHREADS) {
		a = vectsA[bx * dim + offset];
		b = vectsB[by * dim + offset];

		threadSumsA[tx] += a;
		threadSumsB[tx] += b;
		count[tx] += real(1);
	}
	__syncthreads();
    
	for(stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
		if(tx < stride) {
			threadSumsA[tx] += threadSumsA[tx + stride];
			threadSumsB[tx] += threadSumsB[tx + stride];
			count[tx] += count[tx+stride];
		}
		__syncthreads();
	}
	if(tx == 0) {
		means[bx*nb*2+by*2] = threadSumsA[0] / count[0];
		means[bx*nb*2+by*2+1] = threadSumsB[0] / count[0];
		numPairs[bx*nb+by] = count[0];
	}
}

__global__ void gpuSDNoTest(const real * vectsA, size_t na,
	const real * vectsB, size_t nb, size_t dim, 
	const real * means, const real * numPairs, real * sds)
{
	size_t 
		offset, stride,
		tx = threadIdx.x, 
		bx = blockIdx.x, by = blockIdx.y;
	real 
		a, b,
		termA, termB;
	__shared__ real 
		meanA, meanB, n,
		threadSumsA[NUMTHREADS], threadSumsB[NUMTHREADS];

	if((bx >= na) || (by >= nb))
		return;

	if(tx == 0) {
		meanA = means[bx*nb*2+by*2];	
		meanB = means[bx*nb*2+by*2+1];	
		n = numPairs[bx*nb+by]; 
	}
	__syncthreads();

	threadSumsA[tx] = real(0);
	threadSumsB[tx] = real(0);
	for(offset = tx; offset < dim; offset += NUMTHREADS) {
		a = vectsA[bx * dim + offset];
		b = vectsB[by * dim + offset];

		termA = a - meanA;
		termB = b - meanB;
		threadSumsA[tx] += termA * termA;
		threadSumsB[tx] += termB * termB;
	}
	__syncthreads();

	for(stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
		if(tx < stride) {
			threadSumsA[tx] += threadSumsA[tx + stride];
			threadSumsB[tx] += threadSumsB[tx + stride];
		}
		__syncthreads();
	}
	if(tx == 0) {
		sds[bx*nb*2+by*2]   = sqrt(threadSumsA[0] / (n - real(1)));
		sds[bx*nb*2+by*2+1] = sqrt(threadSumsB[0] / (n - real(1)));
	}
}

__global__ void gpuPMCCNoTest(const real * vectsa, size_t na,
	const real * vectsb, size_t nb, size_t dim,
	const real * numPairs, const real * means, const real * sds,
	real * correlations) 
{
	size_t 
		offset, stride,
		x = blockIdx.x, y = blockIdx.y, 
		tx = threadIdx.x;
	real 
		a, b, n, scoreA, scoreB;
    __shared__ real 
		meanA, meanB, 
		sdA, sdB, 
		threadSums[NUMTHREADS];

	if((x >= na) || (y >= nb))
		return;

	if(tx == 0) {
		meanA = means[x*nb*2+y*2];
		meanB = means[x*nb*2+y*2+1];	
		sdA = sds[x*nb*2+y*2];
		sdB = sds[x*nb*2+y*2+1];	
		n = numPairs[x*nb+y]; 
	}
	__syncthreads();

	threadSums[tx] = real(0);
	for(offset = tx; offset < dim; offset += NUMTHREADS) {
		a = vectsa[x * dim + offset];
		b = vectsb[y * dim + offset];
		
		scoreA = (a - meanA) / sdA; 
		scoreB = (b - meanB) / sdB;
		threadSums[tx] += scoreA * scoreB;
	}
	__syncthreads();

	for(stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
		if(tx < stride) threadSums[tx] += threadSums[tx + stride];
		__syncthreads();
	}
	if(tx == 0) correlations[x*nb+y] = threadSums[0] / (n - real(1));
}

__global__ void gpuSignif(const real * gpuNumPairs, 
	const real * gpuCorrelations, size_t n, real * gpuTScores)
{
	size_t 
		i, start,
		bx = blockIdx.x, tx = threadIdx.x;
	real 
		radicand, cor, npairs;

	start = bx * NUMTHREADS * THREADWORK + tx * THREADWORK;
	for(i = 0; i < THREADWORK; i++) {
		if(start+i >= n)
			break;

		// as tScore() in correlation.cpp: +-Inf at |r| = 1, NaN with fewer
		// than three pairs, r clamped to [-1, 1] against rounding
		npairs = gpuNumPairs[start+i];
		cor = fmin(real(1), fmax(real(-1), gpuCorrelations[start+i]));
		radicand = real(1) - cor * cor;
		if(!(npairs > real(2)))
			gpuTScores[start+i] = REAL_NAN;
		else if(radicand <= real(0))
			gpuTScores[start+i] = cor > real(0) ? REAL_INF : -REAL_INF;
		else
			gpuTScores[start+i] = cor * sqrt((npairs - real(2)) / radicand);
	}
}
