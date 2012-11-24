#include<stdio.h>
#include<string.h>
#include<math_constants.h>
#include<cuseful.h>
#include<R.h>
#include<hcluster.h>

#define NUM_THREADS 32
#define NUM_BLOCKS 1024

// Distance matrix
__device__ float * hcluster_dist_d;

// Number of elements in each cluster
__device__ float * hcluster_count_d;

// Arrays for finding the minimum of each row and column containing the minimum
__device__ float * hcluster_min_val_d;
__device__ size_t * hcluster_min_col_d;

// Arrays telling which cluster merged with which cluster
__device__ int * hcluster_sub_d;
__device__ int * hcluster_sup_d;

// Array of the values merged at
__device__ float * hcluster_merge_val_d;

__global__ void convert_kernel(float * dist, size_t pitch_dist, size_t n)
{
  for(size_t index = threadIdx.x; index < n; index += NUM_THREADS) {
    dist[index * pitch_dist + index] = CUDART_INF_F;
  }
}

__global__ void find_min1_kernel(const float * dist, const size_t pitch_dist, 
	const size_t n, const float * count, float * min_val, size_t * min_col, 
	const size_t row_offset)
{
	// Determine which row this block will handle
	const size_t row = row_offset + blockIdx.x;

	// If the row has already been merged, skip the work
	if((threadIdx.x == 0) && (row < n) && (count[row] < 0.f)) {
		min_val[row] = CUDART_INF_F;
		min_col[row] = 0;		
	}

	if((row >= n) || (count[row] <= 0.f))
		return;

	__shared__ float vals[NUM_THREADS];
	__shared__ size_t cols[NUM_THREADS];

	// Initialize with identity
	vals[threadIdx.x] = CUDART_INF_F;
		
	// Find the minimum
	for(size_t col = threadIdx.x; col <= row; col += NUM_THREADS) {
		float t = dist[row * pitch_dist + col];
		if(t < vals[threadIdx.x]) {
			vals[threadIdx.x] = t;
			cols[threadIdx.x] = col;
		}
	}
	__syncthreads();
				
	// Reduce
	for(size_t stride = NUM_THREADS >> 1; stride > 0; stride >>= 1) {
		if((threadIdx.x < stride)
			&& (vals[threadIdx.x] > vals[threadIdx.x + stride]))
		{
			vals[threadIdx.x] = vals[threadIdx.x + stride];
			cols[threadIdx.x] = cols[threadIdx.x + stride];
		}
		__syncthreads();
	}
		
	// Write the result
	if(threadIdx.x == 0) {
		min_val[row] = vals[0];
		min_col[row] = cols[0];
	}
}

__global__ void find_min2_kernel(const float * min_val, const size_t * min_col,
	float * count, int * sub, int * sup, float * val, const size_t n, 
	const size_t iter)
{
	__shared__ float vals[NUM_THREADS];
	__shared__ size_t cols[NUM_THREADS];

	// Initialize with identity
	vals[threadIdx.x] = CUDART_INF_F;
		
	// Find the minimum
	for(size_t row = threadIdx.x; row < n; row += NUM_THREADS) {
		float t = min_val[row];
		if(t < vals[threadIdx.x]) {
			vals[threadIdx.x] = t;
			cols[threadIdx.x] = row;
		}
	}
	__syncthreads();
				
	// Reduce
	for(size_t stride = NUM_THREADS >> 1; stride > 0; stride >>= 1) {
		if(threadIdx.x < stride) {
			if(vals[threadIdx.x] > vals[threadIdx.x + stride]) {
				vals[threadIdx.x] = vals[threadIdx.x + stride];
				cols[threadIdx.x] = cols[threadIdx.x + stride];
			}
		}
		__syncthreads();
	}
	
	// Write out
	if(threadIdx.x == 0) {
		// Winning value is vals[0]
		// Winning row is cols[0]
		// Winning column is min_col[cols[0]]
		int row_winner = cols[0];
		int col_winner = min_col[cols[0]];
		val[iter] = vals[0];
		sub[iter] = col_winner;
		sup[iter] = row_winner;

		count[row_winner] += count[col_winner];
		count[col_winner] *= -1.f;
	}
}

__global__ void single_kernel(float * dist, const size_t pitch_dist, 
	const size_t n, const int * sub, const int * sup, const float * count, 
	const float * val, const size_t iter, const size_t col_offset, 
	const float lambda, const float beta)
{
  size_t col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

  // If it matters
  if(col < n) {
    int col_winner = sub[iter];
    int row_winner = sup[iter];
    float top_val = dist[col_winner * pitch_dist + col];
    float bot_val = dist[row_winner * pitch_dist + col];
    bot_val = min(bot_val, top_val);
    if(col == col_winner || col == row_winner) {
      bot_val = CUDART_INF_F;
    }
    top_val = CUDART_INF_F;
    // Write out
    dist[col_winner * pitch_dist + col] = top_val;
    dist[col * pitch_dist + col_winner] = top_val;
    dist[row_winner * pitch_dist + col] = bot_val;
    dist[col * pitch_dist + row_winner] = bot_val;
  }
}

__global__ void complete_kernel(float * dist, const size_t pitch_dist, 
	const size_t n, const int * sub, const int * sup, const float * count, 
	const float * val, const size_t iter, const size_t col_offset, 
	const float lambda, const float beta)
{
  const size_t col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

  // If it matters
  if(col < n) {

    int 
		col_winner = sub[iter], row_winner = sup[iter];
    float 
		top_val = dist[col_winner * pitch_dist + col], 
		bot_val = dist[row_winner * pitch_dist + col];

    bot_val = fmaxf(bot_val, top_val);
    if((col == col_winner) || (col == row_winner))
		bot_val = CUDART_INF_F;

    top_val = CUDART_INF_F;

    // Write out
    dist[col_winner * pitch_dist + col] = top_val;
    dist[col * pitch_dist + col_winner] = top_val;
    dist[row_winner * pitch_dist + col] = bot_val;
    dist[col * pitch_dist + row_winner] = bot_val;
  }
}

__global__ void wpgma_kernel(float * dist, const size_t pitch_dist, 
	const size_t n, const int * sub, const int * sup, const float * count,  
	const float * val, const size_t iter, const size_t col_offset, 
	const float lambda, const float beta)
{
  const size_t col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

  // If it matters
  if(col < n) {
    int col_winner = sub[iter];
    int row_winner = sup[iter];
    float top_val = dist[col_winner * pitch_dist + col];
    float bot_val = dist[row_winner * pitch_dist + col];
    bot_val = (bot_val + top_val) / 2.0;
    if(col == col_winner || col == row_winner) {
      bot_val = CUDART_INF_F;
    }
    top_val = CUDART_INF_F;
    // Write out
    dist[col_winner * pitch_dist + col] = top_val;
    dist[col * pitch_dist + col_winner] = top_val;
    dist[row_winner * pitch_dist + col] = bot_val;
    dist[col * pitch_dist + row_winner] = bot_val;
  }
}

__global__ void average_kernel(float * dist, const size_t pitch_dist, 
	const size_t n, const int * sub, const int * sup, const float * count, 
	const float * val, const size_t iter, const size_t col_offset, 
	const float lambda, const float beta)
{
  const size_t col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

  // If it matters
  if(col < n) {
    int col_winner = sub[iter];
    int row_winner = sup[iter];
    float top_val = dist[col_winner * pitch_dist + col];
    float bot_val = dist[row_winner * pitch_dist + col];
    float nr = count[row_winner];
    float np = -1.0 * count[col_winner];
    float nq = nr - np;
    bot_val = (top_val * np + bot_val * nq) / nr;
    if(col == col_winner || col == row_winner) {
      bot_val = CUDART_INF_F;
    }
    top_val = CUDART_INF_F;
    // Write out
    dist[col_winner * pitch_dist + col] = top_val;
    dist[col * pitch_dist + col_winner] = top_val;
    dist[row_winner * pitch_dist + col] = bot_val;
    dist[col * pitch_dist + row_winner] = bot_val;
  }
}

__global__ void median_kernel(float * dist, const size_t pitch_dist, 
	const size_t n, const int * sub, const int * sup, const float * count, 
	const float * val, const size_t iter, const size_t col_offset, 
	const float lambda, const float beta)
{
  const size_t col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

  // If it matters
  if(col < n) {
    int col_winner = sub[iter];
    int row_winner = sup[iter];
    float top_val = dist[col_winner * pitch_dist + col];
    float bot_val = dist[row_winner * pitch_dist + col];
    bot_val = (bot_val + top_val) / 2.0 - val[iter] / 4.0;
    if(col == col_winner || col == row_winner) {
      bot_val = CUDART_INF_F;
    }
    top_val = CUDART_INF_F;
    // Write out
    dist[col_winner * pitch_dist + col] = top_val;
    dist[col * pitch_dist + col_winner] = top_val;
    dist[row_winner * pitch_dist + col] = bot_val;
    dist[col * pitch_dist + row_winner] = bot_val;
  }
}

__global__ void mcquitty_kernel(float * dist, const size_t pitch_dist, 
	const size_t n, const int * sub, const int * sup, const float * count, 
	const float * val, const size_t iter, const size_t col_offset, 
	const float lambda, const float beta)
{
  const size_t col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

  // If it matters
  if(col < n) {
    int col_winner = sub[iter];
    int row_winner = sup[iter];
    float top_val = dist[col_winner * pitch_dist + col];
    float bot_val = dist[row_winner * pitch_dist + col];
    bot_val = (bot_val + top_val) / 2.0;
    if(col == col_winner || col == row_winner) {
      bot_val = CUDART_INF_F;
    }
    top_val = CUDART_INF_F;
    // Write out
    dist[col_winner * pitch_dist + col] = top_val;
    dist[col * pitch_dist + col_winner] = top_val;
    dist[row_winner * pitch_dist + col] = bot_val;
    dist[col * pitch_dist + row_winner] = bot_val;
  }
}

__global__ void centroid_kernel(float * dist, size_t pitch_dist, 
	size_t n, const int * sub, const int * sup, const float * count, 
	const float * val, size_t iter, size_t col_offset, 
	float lambda, float beta)
{
	size_t 
		col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

	if(col < n) {		// don't run off the end of the arrays
    	int 
			col_winner = sub[iter], row_winner = sup[iter];
		float 
			top_val = dist[col_winner * pitch_dist + col],
    		bot_val = dist[row_winner * pitch_dist + col],
			nr = count[row_winner], np = -count[col_winner],
			nq = nr - np;

		bot_val = (top_val * np + bot_val * nq)/nr 
			- (np * nq * val[iter])/(nr * nr);
//		bot_val = (nr * (bot_val * np + top_val * nq) - np * nq * val[iter]) 
//			/ (nr * nr);
/*
    float nr = count[row_winner];
    float np = -1.0 * count[col_winner];
    float nq = nr - np;
    bot_val = (top_val * np + bot_val * nq) / nr;
*/	
		if(col == col_winner || col == row_winner)
			bot_val = CUDART_INF_F;
    
		top_val = CUDART_INF_F;

		dist[col_winner * pitch_dist + col] = top_val;
		dist[col * pitch_dist + col_winner] = top_val;
		dist[row_winner * pitch_dist + col] = bot_val;
		dist[col * pitch_dist + row_winner] = bot_val;
	}
}

__global__ void flexible_group_kernel(float * dist, const size_t pitch_dist, 
	const size_t n, const int * sub, const int * sup, const float * count, 
	const float * val, const size_t iter, const size_t col_offset, 
	const float lambda, const float beta)
{
  const size_t col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

  // If it matters
  if(col < n) {
    int col_winner = sub[iter];
    int row_winner = sup[iter];
    float top_val = dist[col_winner * pitch_dist + col];
    float bot_val = dist[row_winner * pitch_dist + col];
    float nr = count[row_winner];
    float np = -1.0 * count[col_winner];
    float nq = nr - np;
    bot_val = (bot_val * (1.0 - lambda) * np + top_val * (1.0 - lambda) *  nq) / nr + beta * val[iter];
    if(col == col_winner || col == row_winner) {
      bot_val = CUDART_INF_F;
    }
    top_val = CUDART_INF_F;
    // Write out
    dist[col_winner * pitch_dist + col] = top_val;
    dist[col * pitch_dist + col_winner] = top_val;
    dist[row_winner * pitch_dist + col] = bot_val;
    dist[col * pitch_dist + row_winner] = bot_val;
  }
}

__global__ void flexible_kernel(float * dist, const size_t pitch_dist, 
	const size_t n, const int * sub, const int * sup, const float * count, 
	const float * val, const size_t iter, const size_t col_offset, 
	const float lambda, const float beta)
{
  const size_t col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

  // If it matters
  if(col < n) {
    int col_winner = sub[iter];
    int row_winner = sup[iter];
    float top_val = dist[col_winner * pitch_dist + col];
    float bot_val = dist[row_winner * pitch_dist + col];
    bot_val = (bot_val * (1.0 - lambda) + top_val * (1.0 - lambda) ) / 2.0 + beta * val[iter];
    if(col == col_winner || col == row_winner) {
      bot_val = CUDART_INF_F;
    }
    top_val = CUDART_INF_F;
    // Write out
    dist[col_winner * pitch_dist + col] = top_val;
    dist[col * pitch_dist + col_winner] = top_val;
    dist[row_winner * pitch_dist + col] = bot_val;
    dist[col * pitch_dist + row_winner] = bot_val;
  }
}

__global__ void ward_kernel(float * dist, const size_t pitch_dist, 
	const size_t n, const int * sub, const int * sup, const float * count, 
	const float * val, const size_t iter, const size_t col_offset, 
	const float lambda, const float beta)
{
	const size_t
		col = col_offset + NUM_THREADS * blockIdx.x + threadIdx.x;

	if(col >= n)
		return;

	int
		col_winner = sub[iter], row_winner = sup[iter];

    float
		top_val = dist[col_winner * pitch_dist + col],
		bot_val = dist[row_winner * pitch_dist + col],
		nr = count[row_winner], np = -count[col_winner],
		nq = nr - np, nk = count[col];

	if((nr == -nk) || (col == col_winner) || (col == row_winner)) {
		bot_val = CUDART_INF_F;
	} else {
		bot_val = (bot_val * (np + nk) + top_val * (nq + nk) - val[iter] * nk);
		bot_val /= (nr + nk);
		if(isinf(bot_val)) {
			bot_val = CUDART_INF_F;
		}
	}
	top_val = CUDART_INF_F;
	
	dist[col_winner * pitch_dist + col] = top_val;
	dist[col * pitch_dist + col_winner] = top_val;
	dist[row_winner * pitch_dist + col] = bot_val;
	dist[col * pitch_dist + row_winner] = bot_val;
}

void hcluster(const float * dist, size_t dist_pitch, size_t n,
	      int * sub, int * sup, float * val, hc_method method,
	      const float lambda, const float beta)
{
	// Initialize the device -- deprecated for CUDA 2.0
	// CUT_DEVICE_INIT(0, NULL);

	// Allocate space for the distance matrix, count array,
	// min_value by row array, min_col by row array,
	// the subordinate array, the superior array, and the merge value.

	size_t pitch_dist_d;

	cudaMallocPitch((void**)&hcluster_dist_d, &pitch_dist_d, n * sizeof(float),
		n);
	cudaMalloc((void**)&hcluster_count_d, n * sizeof(float));
	cudaMalloc((void**)&hcluster_min_val_d, n * sizeof(float));
	cudaMalloc((void**)&hcluster_min_col_d, n * sizeof(size_t));
	cudaMalloc((void**)&hcluster_sub_d, (n - 1) * sizeof(int));
	cudaMalloc((void**)&hcluster_sup_d, (n - 1) * sizeof(int));
	cudaMalloc((void**)&hcluster_merge_val_d, (n - 1) * sizeof(float));

	// Copy the distance matrix
	cudaMemcpy2D(hcluster_dist_d, pitch_dist_d, dist, dist_pitch, 
		n * sizeof(float), n, cudaMemcpyHostToDevice);

	// Every element starts in its own cluster
	float * pre_count = Calloc(n, float);
	for(size_t i = 0; i < n; ++i)
		pre_count[i] = 1.0;

	cudaMemcpy(hcluster_count_d, pre_count, n * sizeof(float), 
		cudaMemcpyHostToDevice);
	checkCudaError("hcluster : malloc and memcpy");

	Free(pre_count);

	dim3 
		grid0(NUM_BLOCKS, 1, 1), block0(NUM_THREADS, 1, 1),
		grid1(1, 1, 1), block1(NUM_THREADS, 1, 1);

	// Convert 0 on the diagonal to infinity
	convert_kernel<<<grid1, block1>>>(hcluster_dist_d, 
		pitch_dist_d / sizeof(float), n);

	checkCudaError("hcluster : convert kernel");

	void (*func)(float *, const size_t, const size_t, const int *, 
		const int *, const float *, const float *, const size_t, 
		const size_t, const float, const float) = NULL;

	switch(method) {
		case COMPLETE:
			func = complete_kernel;
			break;
		case WPGMA:
			func = wpgma_kernel;
			break;
		case AVERAGE:
			func = average_kernel;
			break;
		case MEDIAN:
			func = median_kernel;
			break;
		case CENTROID:
			func = centroid_kernel;
			break;
		case FLEXIBLE_GROUP:
			func = flexible_group_kernel;
			break;
		case FLEXIBLE:
			func = flexible_kernel;
			break;
		case WARD:
			func = ward_kernel;
			break;
		case MCQUITTY:
			func = mcquitty_kernel;
			break;
		case SINGLE:
		default:
			// we have a problem
			func = single_kernel;
			break;
	}

	// Merge items n - 1 times
	for(size_t iter = 0; iter < (n - 1); ++iter) {
		// Find the minimum of each column
		for(size_t row_offset = 0; row_offset < n; row_offset += NUM_BLOCKS) {
			find_min1_kernel<<<grid0, block0>>>(hcluster_dist_d, 
				pitch_dist_d / sizeof(float), n, hcluster_count_d, 
				hcluster_min_val_d, hcluster_min_col_d, row_offset);
		}
		
		// Find overall winner; update arrays sub, sup, val, count
		find_min2_kernel<<<grid1, block1>>>(hcluster_min_val_d, 
			hcluster_min_col_d, hcluster_count_d, hcluster_sub_d, 
			hcluster_sup_d, hcluster_merge_val_d, n, iter);

		// Update the distance matrix
		for(size_t col_offset = 0; col_offset < n; 
			col_offset += NUM_BLOCKS * NUM_THREADS) {

			func<<<grid0, block0>>>(hcluster_dist_d, 
				pitch_dist_d / sizeof(float), n, hcluster_sub_d, 
				hcluster_sup_d, hcluster_count_d, hcluster_merge_val_d,
				iter, col_offset, lambda, beta);
		}
	}
	checkCudaError("hcluster : method kernel calls");

	// Copy results
	cudaMemcpy(sub, hcluster_sub_d, (n - 1) * sizeof(int), 
		cudaMemcpyDeviceToHost);	
	cudaMemcpy(sup, hcluster_sup_d, (n - 1) * sizeof(int), 
		cudaMemcpyDeviceToHost);
	cudaMemcpy(val, hcluster_merge_val_d, (n-1)*sizeof(float), 
		cudaMemcpyDeviceToHost);

	checkCudaError("hcluster : results memcpy");

	cudaFree(hcluster_dist_d);
	cudaFree(hcluster_count_d);
	cudaFree(hcluster_min_val_d);
	cudaFree(hcluster_min_col_d);
	cudaFree(hcluster_sub_d);
	cudaFree(hcluster_sup_d);
	cudaFree(hcluster_merge_val_d);
}

void hclusterPreparedDistances(float * gpuDist, size_t pitch_dist_d, size_t n,
	      int * sub, int * sup, float * val, hc_method method,
	      const float lambda, const float beta)
{
	hcluster_dist_d = gpuDist;

	// Allocate space for the count array,
	// min_value by row array, min_col by row array,
	// the subordinate array, the superior array, and the merge value.

	cudaMalloc((void**)&hcluster_count_d, n * sizeof(float));
	cudaMalloc((void**)&hcluster_min_val_d, n * sizeof(float));
	cudaMalloc((void**)&hcluster_min_col_d, n * sizeof(size_t));
	cudaMalloc((void**)&hcluster_sub_d, (n - 1) * sizeof(int));
	cudaMalloc((void**)&hcluster_sup_d, (n - 1) * sizeof(int));
	cudaMalloc((void**)&hcluster_merge_val_d, (n - 1) * sizeof(float));

	// Every element starts in its own cluster
	float * pre_count = Calloc(n, float);
	for(size_t i = 0; i < n; ++i)
		pre_count[i] = 1.f;

	cudaMemcpy(hcluster_count_d, pre_count, n * sizeof(float), 
		cudaMemcpyHostToDevice);
	checkCudaError("hcluster for gpu distances : initial malloc and memcpy");
	Free(pre_count);

	dim3 
		grid0(NUM_BLOCKS, 1, 1), block0(NUM_THREADS, 1, 1),
		grid1(1, 1, 1), block1(NUM_THREADS, 1, 1);

	// Convert 0 on the diagonal to infinity
	convert_kernel<<<grid1, block1>>>(hcluster_dist_d, 
		pitch_dist_d / sizeof(float), n);
	checkCudaError("hcluster for gpu distances : convert kernel");

	void (*func)(float *, const size_t, const size_t, const int *, 
		const int *, const float *, const float *, const size_t, 
		const size_t, const float, const float) = NULL;

	switch(method) {
		case COMPLETE:
			func = complete_kernel;
			break;
		case WPGMA:
			func = wpgma_kernel;
			break;
		case AVERAGE:
			func = average_kernel;
			break;
		case MEDIAN:
			func = median_kernel;
			break;
		case CENTROID:
			func = centroid_kernel;
			break;
		case FLEXIBLE_GROUP:
			func = flexible_group_kernel;
			break;
		case FLEXIBLE:
			func = flexible_kernel;
			break;
		case WARD:
			func = ward_kernel;
			break;
		case MCQUITTY:
			func = mcquitty_kernel;
			break;
		case SINGLE:
		default:
			// we have a problem
			func = single_kernel;
			break;
	}

	// Merge items n - 1 times
	for(size_t iter = 0; iter < (n - 1); ++iter) {
		// Find the minimum of each column
		for(size_t row_offset = 0; row_offset < n; row_offset += NUM_BLOCKS) {
			find_min1_kernel<<<grid0, block0>>>(hcluster_dist_d, 
				pitch_dist_d / sizeof(float), n, hcluster_count_d, 
				hcluster_min_val_d, hcluster_min_col_d, row_offset);
		}
		
		// Find overall winner; update arrays sub, sup, val, count
		find_min2_kernel<<<grid1, block1>>>(hcluster_min_val_d, 
			hcluster_min_col_d, hcluster_count_d, hcluster_sub_d, 
			hcluster_sup_d, hcluster_merge_val_d, n, iter);

		// Update the distance matrix
		for(size_t col_offset = 0; col_offset < n; 
			col_offset += NUM_BLOCKS * NUM_THREADS) {

			func<<<grid0, block0>>>(hcluster_dist_d, 
				pitch_dist_d / sizeof(float), n, hcluster_sub_d, 
				hcluster_sup_d, hcluster_count_d, hcluster_merge_val_d,
				iter, col_offset, lambda, beta);
		}
	}
	checkCudaError("hcluster for gpu distances : method kernel calls");

	// Copy results
	cudaMemcpy(sub, hcluster_sub_d, (n - 1) * sizeof(int), 
		cudaMemcpyDeviceToHost);	
	cudaMemcpy(sup, hcluster_sup_d, (n - 1) * sizeof(int), 
		cudaMemcpyDeviceToHost);
	cudaMemcpy(val, hcluster_merge_val_d, (n-1)*sizeof(float), 
		cudaMemcpyDeviceToHost);
	checkCudaError("hcluster for gpu distances : results memcpy");

	// Free allocated space
	cudaFree(hcluster_dist_d);
	cudaFree(hcluster_count_d);
	cudaFree(hcluster_min_val_d);
	cudaFree(hcluster_min_col_d);
	cudaFree(hcluster_sub_d);
	cudaFree(hcluster_sup_d);
	cudaFree(hcluster_merge_val_d);
}
