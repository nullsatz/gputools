#define FALSE 0
#define TRUE !FALSE

#define NUMTHREADS 16
#define THREADWORK 32

__global__ void gpuMeans(const float * vectsA, size_t na, 
	const float * vectsB, size_t nb, size_t dim, 
	float * means, float * numPairs)
{
	size_t 
		offset, stride,
		bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;
	float a, b;

	__shared__ float 
		threadSumsA[NUMTHREADS], threadSumsB[NUMTHREADS],
		count[NUMTHREADS];

	if((bx >= na) || (by >= nb))
		return;

	threadSumsA[tx] = 0.f;
	threadSumsB[tx] = 0.f;
	count[tx] = 0.f;

	for(offset = tx; offset < dim; offset += NUMTHREADS) {
		a = vectsA[bx * dim + offset];
		b = vectsB[by * dim + offset];
		if(!(isnan(a) || isnan(b))) {
			threadSumsA[tx] += a;
			threadSumsB[tx] += b;
			count[tx] += 1.f;
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

__global__ void gpuSD(const float * vectsA, size_t na,
	const float * vectsB, size_t nb, size_t dim, 
	const float * means, const float * numPairs, float * sds)
{
	size_t 
		offset, stride,
		tx = threadIdx.x, 
		bx = blockIdx.x, by = blockIdx.y;
	float 
		a, b,
		termA, termB;
	__shared__ float 
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

	threadSumsA[tx] = 0.f;
	threadSumsB[tx] = 0.f;
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
		sds[bx*nb*2+by*2]   = sqrtf(threadSumsA[0] / (n - 1.f));
		sds[bx*nb*2+by*2+1] = sqrtf(threadSumsB[0] / (n - 1.f));
	}
}

__global__ void gpuPMCC(const float * vectsa, size_t na,
	const float * vectsb, size_t nb, size_t dim,
	const float * numPairs, const float * means, const float * sds,
	float * correlations) 
{
	size_t 
		offset, stride,
		x = blockIdx.x, y = blockIdx.y, 
		tx = threadIdx.x;
	float 
		a, b, n, scoreA, scoreB;
    __shared__ float 
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

	threadSums[tx] = 0.f;
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
	if(tx == 0) correlations[x*nb+y] = threadSums[0] / (n - 1.f);
}

__global__ void gpuMeansNoTest(const float * vectsA, size_t na, 
	const float * vectsB, size_t nb, size_t dim, 
	float * means, float * numPairs)
{
	size_t 
		offset, stride,
		bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;
	float a, b;

	__shared__ float 
		threadSumsA[NUMTHREADS], threadSumsB[NUMTHREADS],
		count[NUMTHREADS];

	if((bx >= na) || (by >= nb))
		return;

	threadSumsA[tx] = 0.f;
	threadSumsB[tx] = 0.f;
	count[tx] = 0.f;

	for(offset = tx; offset < dim; offset += NUMTHREADS) {
		a = vectsA[bx * dim + offset];
		b = vectsB[by * dim + offset];

		threadSumsA[tx] += a;
		threadSumsB[tx] += b;
		count[tx] += 1.f;
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

__global__ void gpuSDNoTest(const float * vectsA, size_t na,
	const float * vectsB, size_t nb, size_t dim, 
	const float * means, const float * numPairs, float * sds)
{
	size_t 
		offset, stride,
		tx = threadIdx.x, 
		bx = blockIdx.x, by = blockIdx.y;
	float 
		a, b,
		termA, termB;
	__shared__ float 
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

	threadSumsA[tx] = 0.f;
	threadSumsB[tx] = 0.f;
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
		sds[bx*nb*2+by*2]   = sqrtf(threadSumsA[0] / (n - 1.f));
		sds[bx*nb*2+by*2+1] = sqrtf(threadSumsB[0] / (n - 1.f));
	}
}

__global__ void gpuPMCCNoTest(const float * vectsa, size_t na,
	const float * vectsb, size_t nb, size_t dim,
	const float * numPairs, const float * means, const float * sds,
	float * correlations) 
{
	size_t 
		offset, stride,
		x = blockIdx.x, y = blockIdx.y, 
		tx = threadIdx.x;
	float 
		a, b, n, scoreA, scoreB;
    __shared__ float 
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

	threadSums[tx] = 0.f;
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
	if(tx == 0) correlations[x*nb+y] = threadSums[0] / (n - 1.f);
}

__global__ void gpuSignif(const float * gpuNumPairs, 
	const float * gpuCorrelations, size_t n, float * gpuTScores)
{
	size_t 
		i, start,
		bx = blockIdx.x, tx = threadIdx.x;
	float 
		radicand, cor, npairs;

	start = bx * NUMTHREADS * THREADWORK + tx * THREADWORK;
	for(i = 0; i < THREADWORK; i++) {
		if(start+i >= n)
			break;

		npairs = gpuNumPairs[start+i];
		cor = gpuCorrelations[start+i];
		radicand = (npairs - 2.f) / (1.f - cor * cor);
		gpuTScores[start+i] = cor * sqrtf(radicand);
	}
}
