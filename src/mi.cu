#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<R.h>
#include<cuseful.h>
#include<mi.h>

#define NTHREADS 16

static int initKnots(int nbins, int order, float ** knots) {
	int
		om1 = order - 1,
		degree = nbins - 1,
		dpo = degree + order,
		nknots = dpo + 1;

	*knots = Calloc(nknots, float);	
	for(int i = 0; i < nknots; i++) {
		if(i <= om1)
			(*knots)[i] = 0.f;
		else if(i <= degree)
			(*knots)[i] = (*knots)[i-1] + 1.f;
		else
			(*knots)[i] = (*knots)[degree] + 1.f;
	}
	return nknots;
}

__global__ void scale(float knot_max, int nx, int nsamples, float * x,
	int pitch_x)
{
	int
		col_idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(col_idx >= nx) return;

	float
		min, max,
		* col = x + col_idx * pitch_x;

	// find the min and the max
	min = max = col[0];
	for(int i = 1; i < nsamples; i++) {
		if(col[i] < min) min = col[i];
		if(col[i] > max) max = col[i];
	}

	float delta = max - min;
	for(int i = 0; i < nsamples; i++)
		col[i] = (knot_max * (col[i] - min)) / delta;
}

__device__ float do_fraction(float numer, float denom) {
	float result = 0.f; 

	if((numer == denom) && (numer != 0.f))
		result = 1.f;
	else if(denom != 0.f)
		result = numer / denom;

	return result;
}

// bins must be initialized to zero before calling get_bin_scores
__global__ void get_bin_scores(int nbins, int order,
	int nknots, float * knots, int nsamples, int nx, float * x, int pitch_x,
	float * bins, int pitch_bins)
{
	int
		col_x = blockDim.x * blockIdx.x + threadIdx.x;

	if(col_x >= nx)
		return;

	float
		ld, rd, z,
		term1, term2,
		* in_col = x + col_x * pitch_x,
		* bin_col = bins + col_x * pitch_bins;
	int i0;

	for(int k = 0; k < nsamples; k++, bin_col += nbins) {
		z = in_col[k];
		i0 = (int)floorf(z) + order - 1;
		if(i0 >= nbins)
			i0 = nbins - 1;

		bin_col[i0] = 1.f;
		for(int i = 2; i <= order; i++) {
			for(int j = i0 - i + 1; j <= i0; j++) {
				rd = do_fraction(knots[j + i] - z, knots[j + i] - knots[j + 1]);

				if((j < 0) || (j >= nbins) || (j >= nknots) || (j + i - 1 < 0) || (j > nknots))
					term1 = 0.f;
				else {
					ld = do_fraction(z - knots[j],
						knots[j + i - 1] - knots[j]);
					term1 = ld * bin_col[j];
				}

				if((j + 1 < 0) || (j + 1 >= nbins) || (j + 1 >= nknots) || (j + i < 0) || (j + i >= nknots))
					term2 = 0.f;
				else {
					rd = do_fraction(knots[j + i] - z,
						knots[j + i] - knots[j + 1]);
					term2 = rd * bin_col[j + 1];
				}
				bin_col[j] = term1 + term2;
			}
		}
	}
}

__global__ void get_entropy(int nbins, int nsamples, int nx,
	float * bin_scores, int pitch_bin_scores, float * entropies)
{
	int
		col_x = blockDim.x * blockIdx.x + threadIdx.x;

	if(col_x >= nx)
		return;

	float
		* in_col = bin_scores + col_x * pitch_bin_scores,
		entropy = 0.f, prob, logp;

    for(int i = 0; i < nbins; i++) {
        prob = 0.f;
        for(int j = 0; j < nsamples; j++)
            prob += in_col[j * nbins + i];
        prob /= (double) nsamples;

		if(prob <= 0.f)
			logp = 0.f;
		else
			logp = __log2f(prob);

        entropy += prob * logp;
    }
	entropies[col_x] = -entropy;
}

__global__ void get_mi(int nbins, int nsamples,
	int nx, float * x_bin_scores, int pitch_x_bin_scores, float * entropies_x,
	int ny, float * y_bin_scores, int pitch_y_bin_scores, float * entropies_y,
	float * mis, int pitch_mis)
{
	int
		col_x = blockDim.x * blockIdx.x + threadIdx.x,
		col_y = blockDim.y * blockIdx.y + threadIdx.y;

	if((col_x >= nx) || (col_y >= ny))
		return;

    float
		prob, logp, mi = 0.f,
		* x_bins = x_bin_scores + col_x * pitch_x_bin_scores,
		* y_bins = y_bin_scores + col_y * pitch_y_bin_scores;
		
	// calculate joint entropy
    for(int i = 0; i < nbins; i++) {
        for(int j = 0; j < nbins; j++) {
            prob = 0.f;
            for(int k = 0; k < nsamples; k++)
                prob += x_bins[k * nbins + i] * y_bins[k * nbins + j];
            prob /= (float)nsamples;

			if(prob <= 0.f)
				logp = 0.f;
			else
				logp = __log2f(prob);

            mi += prob * logp;
        }
    }

	// calculate mi from entropies
    mi += entropies_x[col_x] + entropies_y[col_y];
	(mis + col_y * pitch_mis)[col_x] = mi;
}

void bSplineMutualInfo(int nbins, int order, int nsamples, int nx,
	const float * x, int ny, const float * y, float * out_mi)
{
	size_t
		pitch[2], pitch_bins[2],
		col_bytes = (size_t)nsamples * sizeof(float);
	int
		nknots, nblocks[2], size[2] = { nx, ny };
	float
		* knots, * dknots,
		* stage[2], * dx[2], * dentropy[2], * dbins[2];
	const float
		* data[2] = { x, y };

	nknots = initKnots(nbins, order, &knots);
	float knot_max = knots[nknots - 1];
	cudaMalloc((void **)&dknots, nknots * sizeof(float));
	cudaMemcpy(dknots, knots, nknots * sizeof(float), cudaMemcpyHostToDevice);
	Free(knots);

	checkCudaError("bSplineMutualInfoSingle: 1");
	
	for(int i = 0; i < 2; i++) {
		cudaMallocPitch((void **)&(dx[i]), pitch + i, col_bytes, size[i]);
		cudaMallocHost((void **)&(stage[i]), size[i] * col_bytes);
		cudaMalloc((void **)&(dentropy[i]), size[i] * sizeof(float));
		cudaMallocPitch((void **)&(dbins[i]), pitch_bins + i,
			nbins * col_bytes, size[i]);
		cudaMemset2D(dbins[i], pitch_bins[i], 0, nbins * col_bytes, size[i]);

		nblocks[i] = size[i] / NTHREADS;
		if(nblocks[i] * NTHREADS < size[i])
			nblocks[i]++;
	}

	checkCudaError("bSplineMutualInfoSingle: 2");

	cudaStream_t stream[2];
	for(int i = 0; i < 2; i++)
		cudaStreamCreate(stream + i);
	for(int i = 0; i < 2; i++) {
		cudaMemcpyAsync(stage[i], data[i], size[i] * col_bytes,
			cudaMemcpyHostToHost, stream[i]);
		cudaMemcpy2DAsync(dx[i], pitch[i], stage[i], col_bytes, col_bytes,
			size[i], cudaMemcpyHostToDevice, stream[i]);
	}
	for(int i = 0; i < 2; i++) {
		scale<<<nblocks[i], NTHREADS, 0, stream[i]>>>(knot_max, size[i],
			nsamples, dx[i], pitch[i] / sizeof(float));
		get_bin_scores<<<nblocks[i], NTHREADS, 0, stream[i]>>>(nbins, order,
			nknots, dknots, nsamples, size[i], dx[i], pitch[i] / sizeof(float),
			dbins[i], pitch_bins[i] / sizeof(float));
		get_entropy<<<nblocks[i], NTHREADS, 0, stream[i]>>>(nbins, nsamples,
			size[i], dbins[i], pitch_bins[i] / sizeof(float), dentropy[i]);
	}
	checkCudaError("bSplineMutualInfoSingle: 3");

	cudaFree(dknots);
	for(int i = 0; i < 2; i++) {
		cudaFreeHost(stage[i]);
		cudaFree(dx[i]);
	}

	size_t pitch_mi;
	float * dmi;
	cudaMallocPitch((void **)&dmi, &pitch_mi, ny * sizeof(float), nx);

	dim3
		gridDim(nblocks[0], nblocks[1]), blockDim(NTHREADS, NTHREADS);

	get_mi<<<gridDim, blockDim>>>(nbins, nsamples,
		nx, dbins[0], pitch_bins[0] / sizeof(float), dentropy[0],
		ny, dbins[1], pitch_bins[1] / sizeof(float), dentropy[1],
		dmi, pitch_mi / sizeof(float));

	checkCudaError("bSplineMutualInfoSingle: 4");

	for(int i = 0; i < 2; i++)
		cudaFree(dbins[i]);

	float * mi_stage;
	cudaMallocHost((void **)&mi_stage, nx * ny * sizeof(float)); 

	cudaMemcpy2D(mi_stage, ny * sizeof(float), dmi, pitch_mi,
		ny * sizeof(float), nx, cudaMemcpyDeviceToHost);
	checkCudaError("bSplineMutualInfoSingle: 5");
	cudaFree(dmi);

	memcpy(out_mi, mi_stage, nx * ny * sizeof(float));
	cudaFreeHost(mi_stage);
	checkCudaError("bSplineMutualInfoSingle: 6");
}
