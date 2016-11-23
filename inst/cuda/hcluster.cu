#define NUM_THREADS 32
#define NUM_BLOCKS 1024

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

