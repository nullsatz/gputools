#define NUM_THREADS 32

__global__ void euclidean_kernel(const float * vg_a, size_t pitch_a, size_t n_a,
				 const float * vg_b, size_t pitch_b, size_t n_b,
				 size_t k,
				 float * d, size_t pitch_d,
				 float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  // If an element is to be computed
  if(x < n_a && y < n_b) {

    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float t = vg_a[x * pitch_a + offset] - vg_b[y * pitch_b + offset];
      temp[threadIdx.x] += (t * t);
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      d[y * pitch_d + x] = sqrt(temp[0]);
    }
  }
}

__global__ void euclidean_kernel_same(const float * vg_a, size_t pitch_a, 
	size_t n_a, const float * vg_b, size_t pitch_b, size_t n_b,
	size_t k, float * d, size_t pitch_d, float p)
{
	size_t 
		x = blockIdx.x, y = blockIdx.y;

	if((x == y) && (x < n_a) && (threadIdx.x == 0))
		d[y * pitch_d + x] = 0.0;
  
	// If all element is to be computed
	if(y < n_a && x < y) {
		__shared__ float temp[NUM_THREADS];    

		temp[threadIdx.x] = 0.0;
    
		for(size_t offset = threadIdx.x; offset < k; offset += NUM_THREADS) {
			float t = vg_a[x * pitch_a + offset] - vg_a[y * pitch_a + offset];
			temp[threadIdx.x] += (t * t);
		}
    
		// Sync with other threads
		__syncthreads();
    
		// Reduce
		for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
			if(threadIdx.x < stride)
				temp[threadIdx.x] += temp[threadIdx.x + stride];
			__syncthreads();
		}
	    
		// Write to global memory
		if(threadIdx.x == 0) {
			float s = sqrt(temp[0]);
			d[y * pitch_d + x] = s;
			d[x * pitch_d + y] = s;
		}
	}
}

__global__ void maximum_kernel(const float * vg_a, size_t pitch_a, size_t n_a,
			       const float * vg_b, size_t pitch_b, size_t n_b,
			       size_t k,
			       float * d, size_t pitch_d,
			       float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  // If all element is to be computed
  if(x < n_a && y < n_b) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float t = abs(vg_a[x * pitch_a + offset] - vg_b[y * pitch_b + offset]);
      temp[threadIdx.x] = max(temp[threadIdx.x], t);
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] = max(temp[threadIdx.x], temp[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      d[y * pitch_d + x] = temp[0];
    }
  }
}

__global__ void maximum_kernel_same(const float * vg_a, size_t pitch_a, 
	size_t n_a, const float * vg_b, size_t pitch_b, size_t n_b, size_t k,
    float * d, size_t pitch_d, float p)
{
  size_t 
  	x = blockIdx.x, y = blockIdx.y;

  if(x == y && x < n_a && threadIdx.x == 0) {
    d[y * pitch_d + x] = 0.0;
  }

  // If all element is to be computed
  if(y < n_a && x < y) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float t = abs(vg_a[x * pitch_a + offset] - vg_a[y * pitch_a + offset]);
      temp[threadIdx.x] = max(t, temp[threadIdx.x]);
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] = max(temp[threadIdx.x], temp[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      float s = temp[0];
      d[y * pitch_d + x] = s;
      d[x * pitch_d + y] = s;
    }
  }
}

__global__ void manhattan_kernel(const float * vg_a, size_t pitch_a, size_t n_a,
				 const float * vg_b, size_t pitch_b, size_t n_b,
				 size_t k,
				 float * d, size_t pitch_d,
				 float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  // If all element is to be computed
  if(x < n_a && y < n_b) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float t = abs(vg_a[x * pitch_a + offset] - vg_b[y * pitch_b + offset]);
      temp[threadIdx.x] += t;
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      d[y * pitch_d + x] = temp[0];
    }
  }
}

__global__ void manhattan_kernel_same(const float * vg_a, size_t pitch_a, size_t n_a,
				      const float * vg_b, size_t pitch_b, size_t n_b,
				      size_t k,
				      float * d, size_t pitch_d,
				      float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  if(x == y && x < n_a && threadIdx.x == 0) {
    d[y * pitch_d + x] = 0.0;
  }

  // If all element is to be computed
  if(y < n_a && x < y) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float t = abs(vg_a[x * pitch_a + offset] - vg_a[y * pitch_a + offset]);
      temp[threadIdx.x] += t;
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      float s = temp[0];
      d[y * pitch_d + x] = s;
      d[x * pitch_d + y] = s;
    }
  }
}

__global__ void canberra_kernel(const float * vg_a, size_t pitch_a, size_t n_a,
				const float * vg_b, size_t pitch_b, size_t n_b,
				size_t k,
				float * d, size_t pitch_d,
				float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  // If all element is to be computed
  if(x < n_a && y < n_b) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float num = abs(vg_a[x * pitch_a + offset] - vg_b[y * pitch_b + offset]);
      float den = abs(vg_a[x * pitch_a + offset] + vg_b[y * pitch_b + offset]);
      if(den != 0.0) {
	temp[threadIdx.x] += num / den;
      }
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      d[y * pitch_d + x] = temp[0];
    }
  }
}

__global__ void canberra_kernel_same(const float * vg_a, size_t pitch_a, size_t n_a,
				     const float * vg_b, size_t pitch_b, size_t n_b,
				     size_t k,
				     float * d, size_t pitch_d,
				     float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  if(x == y && x < n_a && threadIdx.x == 0) {
    d[y * pitch_d + x] = 0.0;
  }

  // If all element is to be computed
  if(y < n_a && x < y) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float num = abs(vg_a[x * pitch_a + offset] - vg_a[y * pitch_a + offset]);
      float den = abs(vg_a[x * pitch_a + offset] + vg_a[y * pitch_a + offset]);
      if(den != 0.0) {
	temp[threadIdx.x] += num / den;
      }
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      float s = temp[0];
      d[y * pitch_d + x] = s;
      d[x * pitch_d + y] = s;
    }
  }
}

__global__ void binary_kernel(const float * vg_a, size_t pitch_a, size_t n_a,
			      const float * vg_b, size_t pitch_b, size_t n_b,
			      size_t k,
			      float * d, size_t pitch_d,
			      float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  // If all element is to be computed
  if(x < n_a && y < n_b) {
    __shared__ float temp[2 * NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    temp[threadIdx.x + NUM_THREADS] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      int a = vg_a[x * pitch_a + offset] != 0.0;
      int b = vg_b[y * pitch_b + offset] != 0.0;
      if(a ^ b) {
	temp[threadIdx.x] += 1.0;
      }
      if(a || b) {
	temp[threadIdx.x + NUM_THREADS] += 1.0;
      }
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
	temp[threadIdx.x + NUM_THREADS] += temp[threadIdx.x + stride + NUM_THREADS];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      float val = temp[0];
      if(temp[NUM_THREADS] != 0.0) {
	val /= temp[NUM_THREADS];
      }
      d[y * pitch_d + x] = val;
    }
  }
}

__global__ void binary_kernel_same(const float * vg_a, size_t pitch_a, 
	size_t n_a, const float * vg_b, size_t pitch_b, size_t n_b, size_t k,
   float * d, size_t pitch_d, float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  if(x == y && x < n_a && threadIdx.x == 0) {
    d[y * pitch_d + x] = 0.0;
  }

  // If all element is to be computed
  if(y < n_a && x < y) {
    __shared__ float temp[2 * NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    temp[threadIdx.x + NUM_THREADS] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      int a = vg_a[x * pitch_a + offset] != 0.0;
      int b = vg_a[y * pitch_a + offset] != 0.0;
      if(a ^ b) {
	temp[threadIdx.x] += 1.0;
      }
      if(a || b) {
	temp[threadIdx.x + NUM_THREADS] += 1.0;
      }
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
	temp[threadIdx.x + NUM_THREADS] += temp[threadIdx.x + stride + NUM_THREADS];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      float val = temp[0];
      if(temp[NUM_THREADS] != 0.0) {
	val /= temp[NUM_THREADS];
      }
      d[y * pitch_d + x] = val;
      d[x * pitch_d + y] = val;
    }
  }
}

__global__ void minkowski_kernel(const float * vg_a, size_t pitch_a, size_t n_a,
				 const float * vg_b, size_t pitch_b, size_t n_b,
				 size_t k,
				 float * d, size_t pitch_d,
				 float p)
{
	size_t 
		x = blockIdx.x, y = blockIdx.y;

  // If all element is to be computed
  if(x < n_a && y < n_b) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float t = fabsf(vg_a[x * pitch_a + offset] - vg_b[y * pitch_b + offset]);
      temp[threadIdx.x] += __powf(t, p);
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
		for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
			if(threadIdx.x < stride)
				temp[threadIdx.x] += temp[threadIdx.x + stride];
			__syncthreads();
		}
    // Write to global memory
		if(threadIdx.x == 0) {
			float power = 1.f/p;
			d[y * pitch_d + x] = __powf(temp[0], power);
		}
  }
}

__global__ void minkowski_kernel_same(const float * vg_a, size_t pitch_a, 
	size_t n_a, const float * vg_b, size_t pitch_b, size_t n_b, size_t k, 
	float * d, size_t pitch_d, float p)
{
	size_t x = blockIdx.x;
	size_t y = blockIdx.y;

  if(x == y && x < n_a && threadIdx.x == 0) {
    d[y * pitch_d + x] = 0.0;
  }

  // If all element is to be computed
  if(y < n_a && x < y) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float t = fabsf(vg_a[x * pitch_a + offset] - vg_a[y * pitch_a + offset]);
      temp[threadIdx.x] += __powf(t, p);
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
		float 
			power = 1.f / p, s = __powf(temp[0], power);

      d[y * pitch_d + x] = s;
      d[x * pitch_d + y] = s;
    }
  }
}

__global__ void dot_kernel(const float * vg_a, size_t pitch_a, size_t n_a,
			   const float * vg_b, size_t pitch_b, size_t n_b,
			   size_t k,
			   float * d, size_t pitch_d,
			   float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  // If all element is to be computed
  if(x < n_a && y < n_b) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float t = vg_a[x * pitch_a + offset] * vg_b[y * pitch_b + offset];
      temp[threadIdx.x] += t;
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      d[y * pitch_d + x] = temp[0];
    }
  }
}

__global__ void dot_kernel_same(const float * vg_a, size_t pitch_a, size_t n_a,
	const float * vg_b, size_t pitch_b, size_t n_b,
	size_t k,
	float * d, size_t pitch_d,
	float p)
{
  size_t x = blockIdx.x;
  size_t y = blockIdx.y;

  // If all element is to be computed
  if(y < n_a && x <= y) {
    __shared__ float temp[NUM_THREADS];

    temp[threadIdx.x] = 0.0;
    for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
      float t = vg_a[x * pitch_a + offset] * vg_a[y * pitch_a + offset];
      temp[threadIdx.x] += t;
    }
    
    // Sync with other threads
    __syncthreads();
    
    // Reduce
    for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if(threadIdx.x < stride) {
	temp[threadIdx.x] += temp[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // Write to global memory
    if(threadIdx.x == 0) {
      d[y * pitch_d + x] = temp[0];
      d[x * pitch_d + y] = temp[0];
    }
  }
}
