#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cublas.h>
#include<cuseful.h>
#include<R.h>
#include<correlation.h>

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

__host__ void testSignif(const float * goodPairs, const float * coeffs, 
	size_t n, float * tscores)
{
	size_t
		fbytes = sizeof(float), size = n*fbytes;
	float
		* gpuPairs, * gpuCoeffs, * gpuTs;


	cudaMalloc((void **)&gpuPairs, size);
	cudaMalloc((void**)&gpuCoeffs, size);
	cudaMalloc((void**)&gpuTs, size);

	cudaMemcpy(gpuPairs, goodPairs, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuCoeffs, coeffs, size, cudaMemcpyHostToDevice);

	size_t nblocks = n / (NUMTHREADS*THREADWORK);
	if(nblocks*(NUMTHREADS*THREADWORK) < n) nblocks++;
	dim3 
		tblock(NUMTHREADS), tgrid(nblocks);

	gpuSignif<<<tgrid, tblock>>>(gpuPairs, gpuCoeffs, n, gpuTs);

	cudaMemcpy(tscores, gpuTs, size, cudaMemcpyDeviceToHost);
	cudaFree(gpuPairs);
	cudaFree(gpuCoeffs);
	cudaFree(gpuTs);
}

void hostSignif(const float * goodPairs, const float * coeffs, size_t n, 
	float * tscores)
{
	float cor, radicand;
	for(size_t i = 0; i < n; i++) {
		cor = coeffs[i];
		if(cor >= 0.999f) 
			tscores[i] = 10000.0f;
		else {
			radicand = (goodPairs[i] - 2.f) / (1.f - cor * cor);
			tscores[i] = cor * sqrtf(radicand);
		}
	}
}

__host__ void pmcc(UseObs whichObs, const float * vectsa, size_t na,
	const float * vectsb, size_t nb, size_t dim, float * numPairs, 
	float * correlations, float * signifs)
{
	size_t 
		fbytes = sizeof(float);
	int 
		same = (vectsa == vectsb);
	float
		* gpuVA, * gpuVB,
		* gpuNumPairs, * gpumean, * gpuSds, 
		* gpuCorrelations;
	dim3 
		block(NUMTHREADS), grid(na, nb);

	cudaMalloc((void **)&gpuNumPairs, na*nb*fbytes);
	cudaMalloc((void **)&gpumean, na*nb*2*fbytes);
	cudaMalloc((void **)&gpuSds, na*nb*2*fbytes);
	cudaMalloc((void**)&gpuCorrelations, na*nb*fbytes);

	cudaMalloc((void**)&gpuVA, na*dim*fbytes);
	cudaMemcpy(gpuVA, vectsa, na*dim*fbytes, cudaMemcpyHostToDevice);

	if(!same) { 
		cudaMalloc((void**)&gpuVB, nb*dim*fbytes);
		cudaMemcpy(gpuVB, vectsb, nb*dim*fbytes, cudaMemcpyHostToDevice);
	} else {
		gpuVB = gpuVA;
	}
	checkCudaError("PMCC function : malloc and memcpy");

	switch(whichObs) {
		case pairwiseComplete:
			gpuMeans<<<grid, block>>>(gpuVA, na, gpuVB, nb, dim, gpumean,
				gpuNumPairs);
			cudaThreadSynchronize();
			gpuSD<<<grid, block>>>(gpuVA, na, gpuVB, nb, dim, gpumean, gpuNumPairs, 
				gpuSds);
			cudaThreadSynchronize();
			gpuPMCC<<<grid, block>>>(gpuVA, na, gpuVB, nb, dim, gpuNumPairs,
				gpumean, gpuSds, gpuCorrelations); 
			break;
		default:
			gpuMeansNoTest<<<grid, block>>>(gpuVA, na, gpuVB, nb, dim, gpumean,
				gpuNumPairs);
			cudaThreadSynchronize();
			gpuSDNoTest<<<grid, block>>>(gpuVA, na, gpuVB, nb, dim, gpumean,
				gpuNumPairs, gpuSds);
			cudaThreadSynchronize();
			gpuPMCCNoTest<<<grid, block>>>(gpuVA, na, gpuVB, nb, dim, gpuNumPairs,
				gpumean, gpuSds, gpuCorrelations); 
	}

	cudaMemcpy(correlations, gpuCorrelations, na*nb*fbytes, 
		cudaMemcpyDeviceToHost);
	cudaMemcpy(numPairs, gpuNumPairs, na*nb*fbytes, cudaMemcpyDeviceToHost);
	checkCudaError("PMCC function : kernel finish and memcpy");

	hostSignif(numPairs, correlations, na*nb, signifs);
    
	// Free allocated space
	cudaFree(gpuNumPairs);
	cudaFree(gpumean);
	cudaFree(gpuSds);
	cudaFree(gpuCorrelations);
	cudaFree(gpuVA);
	if(!same) {
		cudaFree(gpuVB);
	}
}

void setDevice(int device) {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if((device < 0) || (device >= deviceCount))
		fatal("The gpu id number is not valid.");

	cudaSetDevice(device);
	checkCudaError("setDevice function : choosing gpu");
}

void getDevice(int * device) {
	cudaGetDevice(device);
}

void findMinMax(const int * data, size_t n, int * minVal, int * maxVal) {
	size_t i;
	*minVal = *maxVal = data[0];
	for(i = 1; i < n; i++) {
		if(data[i] < *minVal) *minVal = data[i];
		if(data[i] > *maxVal) *maxVal = data[i];
	}
}

void getData(const int * images, 
	const int * xcoords, const int * ycoords, const int * zcoords,
	const int * mins, const int * maxes,
	const float * evs, size_t numrows, size_t numimages, float * output)
{
	int 
		xmin, ymin, zmin,
		xmax, ymax, zmax,
		prevImg;
	size_t
		i, j = 0,
		nx, ny, nz,
		x, y, z;

	xmin = mins[0];
	ymin = mins[1];
	zmin = mins[2];	

	xmax = maxes[0];
	ymax = maxes[1];
	zmax = maxes[2];	

	nx = 1+abs(xmax - xmin);
	ny = 1+abs(ymax - ymin);
	nz = 1+abs(zmax - zmin);

	prevImg = images[0];
	for(i = 0; (i < numrows) && (j < numimages); i++) {
		if(prevImg != images[i]) {
			prevImg = images[i];
			j++;
		}
		x = xcoords[i] - xmin;
		y = ycoords[i] - ymin;
		z = zcoords[i] - zmin;
		output[j*nx*ny*nz+x*ny*nz+y*nz+z] = evs[i];
	}
}

size_t parseResults(const int * imageList1, size_t numImages1, 
	const int * imageList2, size_t numImages2,
	int structureid,
	double cutCorrelation, int cutPairs,
	const double * correlations, const double * signifs, const int * numPairs, 
	double * results)
{
	size_t 
		i, j, pos, nrows = 0,
		readpos, writepos;
	double 
		sid = (double) structureid, 
		npair, img1, img2, coeff, signif;

	for(i = 0; i < numImages1; i++) {
		pos = i*numImages2;
		img1 = (double) imageList1[i];
		for(j = 0; j < numImages2; j++) {
			readpos = pos + j;

			npair = (double) numPairs[readpos];
			coeff = (double) correlations[readpos];
			signif = (double) signifs[readpos];
			img2 = (double) imageList2[j];

			if((fabs(coeff) >= cutCorrelation) && (npair >= cutPairs) 
				&& (img1 < img2) && isSignificant(signif, numPairs[readpos]-2))
			{
				writepos = 6*nrows;
				results[writepos]   = img1;
				results[writepos+1] = img2;
				results[writepos+2] = sid;
				results[writepos+3] = coeff;
				results[writepos+4] = signif;
				results[writepos+5] = npair;
				nrows++;
			}
		}
	}
	return nrows;
}

int isSignificant(double signif, int df) {
	double tcutoffs[49] = {
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

	if(fabs(signif) < tcutoffs[index]) return FALSE;

	return TRUE;
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

size_t signifFilter(const double * data, size_t rows, double * results)
{
	size_t i, inrow, outrow, rowcount = 0;
	double tscore, radicand, cor, npairs;

	if(results == NULL) {
		fprintf(stderr, "signifFilter : no ram set aside for results\n");
		exit(EXIT_FAILURE);
	}

	for(i = 0; i < rows; i++) {
		inrow = i * 5;
		outrow = rowcount * 6;	

		cor = data[inrow+3];
		npairs = data[inrow+4];

		if(cor >= 0.999) 
			tscore = 10000.0;
		else {
			radicand = (npairs - 2.0) / (1.0 - cor * cor);
			tscore = cor * sqrt(radicand);
		}
		if(isSignificant(tscore, (int)(npairs-2))) {
			results[outrow]   = data[inrow];
			results[outrow+1] = data[inrow+1];
			results[outrow+2] = data[inrow+2];
			results[outrow+3] = cor;
			results[outrow+4] = tscore;
			results[outrow+5] = npairs;
			rowcount++;
		}
	}
	return rowcount;
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

__host__ void updateSignifs(const float * data, size_t n, float * results)
{
	size_t
		fbytes = sizeof(float), size = n*fbytes,
		insize = 5*size, outsize = 6*size;
	float
		* gpuData, * gpuResults;

	cudaMalloc((void **)&gpuData, insize);
	cudaMalloc((void**)&gpuResults, outsize);
	checkCudaError("updateSignifs function : gpu out of RAM");

	cudaMemcpy(gpuData, data, insize, cudaMemcpyHostToDevice);

	size_t nblocks = n / (NUMTHREADS*THREADWORK);
	if(nblocks*(NUMTHREADS*THREADWORK) < n) nblocks++;
	dim3 
		tblock(NUMTHREADS), tgrid(nblocks);

	dUpdateSignif<<<tgrid, tblock>>>(gpuData, n, gpuResults);
	checkCudaError("updateSignifs function : trouble executing kernel on gpu");

	cudaMemcpy(results, gpuResults, outsize, cudaMemcpyDeviceToHost);
	cudaFree(gpuData);
	cudaFree(gpuResults);
	checkCudaError("updateSignifs function : trouble reading from gpu");
}

size_t gpuSignifFilter(const float * data, size_t rows, float * results)
{
	size_t 
		i, rowbytes = 6*sizeof(float),
		inrow, outrow, rowcount = 0;

	if(results == NULL) {
		fprintf(stderr, "signifFilter : no ram set aside for results\n");
		exit(EXIT_FAILURE);
	}

	updateSignifs(data, rows, results);

	for(i = 0; i < rows; i++) {
		inrow = i*6;
		outrow = rowcount*6;	

		if(results[inrow] == -1.f) continue;

		memcpy(results+outrow, results+inrow, rowbytes); 
		rowcount++;
	}
	return rowcount;
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

void cublasPMCC(const float * sampsa, size_t numSampsA, const float * sampsb, 
	size_t numSampsB, size_t sampSize, float * res)
{
	int 
		same = (sampsa == sampsb);
	float
		* gpua, * gpub, * gpuRes,
		* aRecipSD = Calloc(numSampsA, float);

	cublasInit();
	cublasAlloc(numSampsA*sampSize, sizeof(float), (void **)&gpua);

	checkCublasError("PMCC : alloc for input A");
	cublasSetVector(numSampsA*sampSize, sizeof(float), sampsa, 1, gpua, 1);

	float * gpuaRecipSD;
	cublasAlloc(numSampsA, sizeof(float), (void **)&gpuaRecipSD);

	dim3
		dimBlock(NUMTHREADS, NUMTHREADS);

	int numBlocks = numSampsA / NUMTHREADS;
	if(numBlocks * NUMTHREADS < numSampsA)
		numBlocks++;

	checkCublasError("PMCC : alloc and set workspace for A");

	for(size_t i = 0; i < sampSize; i++) // subtract mean from each sample
		cublasSaxpy(numSampsA, -1.f, gpuaRecipSD, 1, gpua+i, sampSize);
	cublasFree(gpuaRecipSD);

	for(size_t i = 0; i < numSampsA; i++) { // sum of squares
		float denom = 
			cublasSdot(sampSize, gpua+i*sampSize, 1, gpua+i*sampSize, 1);
		aRecipSD[i] = 1.f / sqrtf(denom);
	}
	for(size_t i = 0; i < numSampsA; i++) // div each sample by sqrt of s-o-s
		cublasSscal(sampSize, aRecipSD[i], gpua+i*sampSize, 1); 
	Free(aRecipSD);
	checkCublasError("PMCC : vector ops for A");

	if(!same) {
		float * bRecipSD = Calloc(numSampsB, float);
	
		cublasAlloc(numSampsB*sampSize, sizeof(float), (void **)&gpub);
		cublasSetVector(numSampsB*sampSize, sizeof(float), sampsb, 1, gpub, 1);

		float * gpubRecipSD;
		cublasAlloc(numSampsB, sizeof(float), (void **)&gpubRecipSD);

		dim3 dimBlock(NUMTHREADS, NUMTHREADS);
		int numBlocks = numSampsB / NUMTHREADS;
		if(numBlocks * NUMTHREADS < numSampsB)
			numBlocks++;

		noNAsPmccMeans<<<numBlocks, dimBlock>>>(sampSize, numSampsB, gpub, 
			gpubRecipSD);
	
		for(size_t i = 0; i < sampSize; i++) // subtract mean
			cublasSaxpy(numSampsB, -1.f, gpubRecipSD, 1, gpub+i, sampSize);
		cublasFree(gpubRecipSD);
	
		for(size_t i = 0; i < numSampsB; i++) { // sum of squares
			float denom = 
				cublasSdot(sampSize, gpub+i*sampSize, 1, 
					gpub+i*sampSize, 1);
			bRecipSD[i] = 1.f / sqrtf(denom);
		}
		for(size_t i = 0; i < numSampsB; i++) // div by s-o-s
			cublasSscal(sampSize, bRecipSD[i], gpub+i*sampSize, 1); 
		Free(bRecipSD);
		checkCublasError("PMCC : setup for matrix B");
	} else {
		gpub = gpua;
	}

	cublasAlloc(numSampsA*numSampsB, sizeof(float), (void **)&gpuRes);
	cublasSgemm('T', 'N', numSampsB, numSampsA, sampSize, 1.f, 
		gpub, sampSize, gpua, sampSize, 0.f, gpuRes, numSampsB); 
		// each entry : sum of prod of standard scores
	cublasGetVector(numSampsA*numSampsB, sizeof(float), gpuRes, 1, res, 1);
	checkCublasError("PMCC : alloc, matrix mult and get result");
	cublasFree(gpuRes);
	cublasShutdown();
}

double hostKendall(const float * X, const float * Y, size_t n) {
	float concordant, discordant, denom;

	concordant = discordant = 0.0;
	for(size_t i = 0; i < n; i++) {
		for(size_t j = i+1; j < n; j++) {
			if((X[j] > X[i]) && (Y[j] > Y[i])) 
				concordant = concordant + 1.0;
			else if((X[j] < X[i]) && (Y[j] < Y[i])) 
				concordant = concordant + 1.0;
		}
	}
	denom = (double) n;
	denom = denom*(denom - 1.0) / 2.0;
	discordant = denom - concordant;
	return (double)((double)(concordant - discordant)) / ((double)denom);
}

void permHostKendall(const float * a, size_t na, const float * b, size_t nb,
	size_t sampleSize, double * results)
{
	for(size_t i = 0; i < nb; i++) {
		for(size_t j = 0; j < na; j++) {
			results[i*na+j] = hostKendall(a+j*sampleSize, b+i*sampleSize, 
				sampleSize);
		}
	}
}
