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

__device__ int dIsSignificant(float signif, int df) {
  float tcutoffs[49] = {
    // cuttoffs for degrees of freedom <= 30
    637.000, 31.600, 2.920, 8.610, 6.869, 5.959, 5.408, 5.041, 4.781, 
    4.587, 4.437, 4.318, 4.221, 4.140, 4.073, 4.015, 3.965, 3.922, 
    3.883, 3.850, 3.819, 3.792, 3.768, 3.745, 3.725, 3.707, 3.690, 
    3.674, 3.659, 3.646,
    // cuttoffs for even degrees of freedom > 30 but <= 50
    3.622, 3.601, 3.582, 3.566, 3.551, 3.538, 3.526, 3.515, 3.505, 3.496, 
    // 55 <= df <= 70 by 5s
    3.476, 3.460, 3.447, 3.435, 
    3.416, // 80
    3.390, // 100
    3.357, // 150
    3.340, // 200
    3.290  // > 200
  };

  size_t index = 0;
  if(df <= 0) return 0;
  else if(df <= 30) index = df - 1;
  else if(df <= 50) index = 30 + (df + (df%2) - 32) / 2;
  else if(df <= 70) {
    if(df <= 55) index = 40;
    else if(df <= 60) index = 41;
    else if(df <= 65) index = 42;
    else if(df <= 70) index = 43;
  }
  else if(df <= 80) index = 44;
  else if(df <= 100) index = 45;
  else if(df <= 150) index = 46;
  else if(df <= 200) index = 47;
  else if(df > 200) index = 48;

  if(fabsf(signif) < tcutoffs[index]) return FALSE;

  return TRUE;
}

__global__ void dUpdateSignif(const float * gpuData, size_t n, 
                              float * gpuResults)
{
  size_t 
    i, start, inrow, outrow, 
    bx = blockIdx.x, tx = threadIdx.x;
  float 
    radicand, cor, npairs, tscore;

  start = bx * NUMTHREADS * THREADWORK + tx * THREADWORK;
        
  for(i = 0; i < THREADWORK; i++) {
    if(start+i > n) break;

    inrow = (start+i)*5;
    outrow = (start+i)*6;

    cor = gpuData[inrow+3];
    npairs = gpuData[inrow+4];

    if(cor >= 0.999) 
      tscore = 10000.0;
    else {
      radicand = (npairs - 2.f) / (1.f - cor * cor);
      tscore = cor * sqrtf(radicand);
    }
    if(dIsSignificant(tscore, (int)npairs)) {
      gpuResults[outrow]   = gpuData[inrow];
      gpuResults[outrow+1] = gpuData[inrow+1];
      gpuResults[outrow+2] = gpuData[inrow+2];
      gpuResults[outrow+3] = cor;
      gpuResults[outrow+4] = tscore;
      gpuResults[outrow+5] = npairs;
    } else {
      gpuResults[outrow] = -1.f;
    }
  }
}

__global__ void noNAsPmccMeans(int nRows, int nCols, float * a, float * means)
{
	int
		col = blockDim.x * blockIdx.x + threadIdx.x,
		inOffset = col * nRows,
		outOffset = threadIdx.x * blockDim.y,
		j = outOffset + threadIdx.y;
	float sum = 0.f;

	if(col >= nCols) return;

	__shared__ float threadSums[NUMTHREADS*NUMTHREADS];

	for(int i = threadIdx.y; i < nRows; i += blockDim.y)
		sum += a[inOffset + i];

	threadSums[j] = sum;
	__syncthreads();

	for(int i = blockDim.y >> 1; i > 0; i >>= 1) {
		if(threadIdx.y < i) {
			threadSums[outOffset+threadIdx.y] 
				+= threadSums[outOffset+threadIdx.y + i];
		}
		__syncthreads();
	}
	if(threadIdx.y == 0)
		means[col] = threadSums[outOffset] / (float)nRows;
}
