#include<stdio.h>
#include<string.h>
#include<cuseful.h>
#include<distance.h>

#define NUM_THREADS 32

// Space for the vector data
__constant__ float * distance_vg_a_d;
__constant__ float * distance_vg_b_d;

// Space for the resulting distance
__device__ float * distance_d_d;

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

static void euclidean(const float * vg_a, size_t n_a, const float * vg_b, 
	size_t n_b, size_t dim, float * d)
{
	float 
		sum, component;

	for(size_t y = 0; y < n_b; y++) {
		for(size_t x = 0; x < n_a; x++) {
			sum = 0.f;
			for(size_t i = 0; i < dim; i++) {
				component = vg_a[x * dim + i] - vg_b[y * dim + i];
				sum += component * component;
			}
			d[y * dim + x] = sqrtf(sum);
		}
	}
}

static void maximum(const float * vg_a, size_t n_a, const float * vg_b, 
	size_t n_b, size_t dim, float * d)
{
	float 
		themax, previous, current;

	for(size_t y = 0; y < n_b; y++) {
		for(size_t x = 0; x < n_a; x++) {
			previous = 0.f;
			for(size_t i = 0; i < dim; i++) {
				current = fabsf(vg_a[x * dim + i] - vg_b[y * dim + i]);
				themax = (previous < current)? current : previous;
				previous = themax;
			}
			d[y * dim + x] = themax;
		}
	}
}

static void manhattan(const float * vg_a, size_t n_a, const float * vg_b, 
	size_t n_b, size_t dim, float * d)
{
	float sum;

	for(size_t y = 0; y < n_b; y++) {
		for(size_t x = 0; x < n_a; x++) {
			sum = 0.f;
			for(size_t i = 0; i < dim; i++)
				sum += fabsf(vg_a[x * dim + i] - vg_b[y * dim + i]);
			d[y * dim + x] = sum;
		}
	}
}

static void canberra(const float * vg_a, size_t n_a, const float * vg_b, 
	size_t n_b, size_t dim, float * d)
{
	float
		componentDiff, componentSum,
		acoord, bcoord,
		sum;

	for(size_t y = 0; y < n_b; y++) {
		for(size_t x = 0; x < n_a; x++) {
			sum = 0.f;
			for(size_t i = 0; i < dim; i++) {
				acoord = vg_a[x * dim + i];
				bcoord = vg_b[y * dim + i];

				componentDiff = fabsf(acoord - bcoord);
				componentSum = fabsf(acoord + bcoord);

				if(componentSum != 0.f) 
					sum += componentDiff / componentSum;
			}
			d[y * dim + x] = sum;
		}
	}
}

static void binary(const float * vg_a, size_t n_a, const float * vg_b, 
	size_t n_b, size_t dim, float * d)
{
	int
		acoord, bcoord,
		sharedOnes, ones;
	float ratio;

	for(size_t y = 0; y < n_b; ++y) {
		for(size_t x = 0; x < n_a; ++x) {
			ones = sharedOnes = 0;
			for(size_t i = 0; i < dim; ++i) {
				acoord = (vg_a[x * dim + i] != 0.f);
				bcoord = (vg_b[y * dim + i] != 0.f);

				if(acoord ^ bcoord) sharedOnes += 1;
				if(acoord || bcoord) ones += 1;
			}
			ratio = (ones != 0)? ((float)sharedOnes / (float)ones) 
				: (float)sharedOnes;
				
			d[y * dim + x] = ratio;
		}
    }
}

static void minkowski(const float * vg_a, size_t n_a, const float * vg_b, 
	size_t n_b, size_t dim, float p, float * d)
{
	float
		component, sum;

	for(size_t y = 0; y < n_b; y++) {
		for(size_t x = 0; x < n_a; x++) {
			sum = 0.f;
			for(size_t i = 0; i < dim; i++) {
				component = fabsf(vg_a[x * dim + i] - vg_b[y * dim + i]);
				sum += powf(component, p);
			}
			d[y * dim + x] = powf(sum, (float)(1.f / p));
		}
	}
}

/*
static void dot(const float * vg_a, size_t pitch_a, size_t n_a,
		const float * vg_b, size_t pitch_b, size_t n_b,
		size_t k,
		float * d, size_t pitch_d)
{
  // Two different vectors
  if(vg_a != vg_b) {
    for(size_t y = 0; y < n_b; ++y) {
      for(size_t x = 0; x < n_a; ++x) {
	float s = 0.0;
	for(size_t i = 0; i < k; ++i) {
	  float t = vg_a[x * pitch_a + i] * vg_b[y * pitch_b + i];
	  s += t;
	}
	d[y * pitch_d + x] = s;
      }
    }
  } else {
    // Compute
    for(size_t y = 1; y < n_b; ++y) {
      for(size_t x = 0; x <= y; ++x) {
	float s = 0.0;
	for(size_t i = 0; i < k; ++i) {
	  float t = vg_a[x * pitch_a + i] * vg_b[y * pitch_b + i];
	  s += t;
	}
	d[y * pitch_d + x] = s;
	d[x * pitch_d + y] = s;
      }
    }
  }
}
*/

void distance_host(const float * vg_a, size_t pitch_a, size_t n_a,
		   const float * vg_b, size_t pitch_b, size_t n_b,
		   size_t k, float * d, size_t pitch_d,
		   dist_method method, float p)
{
	switch(method) {
		case EUCLIDEAN:
			euclidean(vg_a, n_a, vg_b, n_b, k, d);
			break;
		case MAXIMUM:
			maximum(vg_a, n_a, vg_b, n_b, k, d);
			break;
		case MANHATTAN:
			manhattan(vg_a, n_a, vg_b, n_b, k, d);
			break;
		case CANBERRA:
			canberra(vg_a, n_a, vg_b, n_b, k, d);
			break;
		case BINARY:
			binary(vg_a, n_a, vg_b, n_b, k, d);
			break;
		case MINKOWSKI:
			minkowski(vg_a, n_a, vg_b, n_b, k, p, d);
			break;
		default:
			fprintf(stderr, "unknown distance method");
			exit(EXIT_FAILURE);
/*  case DOT:
    dot(vg_a, pitch_a / sizeof(float), n_a,
	vg_b, pitch_b / sizeof(float), n_b,
	k,
	d, pitch_d / sizeof(float));
    break; */
	}
}

void distance_device(const float * vg_a_d, size_t pitch_a, size_t n_a,
	const float * vg_b_d, size_t pitch_b, size_t n_b,
	size_t k, float * d_d, size_t pitch_d, dist_method method, float p)
{
	dim3 block(NUM_THREADS, 1, 1);
	dim3 grid(n_a, n_b, 1);

	size_t fbytes = sizeof(float);

	pitch_a /= fbytes;
	pitch_b /= fbytes;
	pitch_d /= fbytes;
  
	switch(method) {  // Calculate the distance
		case EUCLIDEAN:
			euclidean_kernel_same<<<grid, block>>>(vg_a_d, pitch_a, n_a, 
				vg_b_d, pitch_b, n_b, k, d_d, pitch_d, p);
			break;
		case MAXIMUM:
			maximum_kernel_same<<<grid, block>>>(vg_a_d, pitch_a, n_a, 
				vg_b_d, pitch_b, n_b, k, d_d, pitch_d, p);
			break;
		case MANHATTAN:
			manhattan_kernel_same<<<grid, block>>>(vg_a_d, pitch_a, n_a, 
				vg_b_d, pitch_b, n_b, k, d_d, pitch_d, p);
			break;
		case CANBERRA:
			canberra_kernel_same<<<grid, block>>>(vg_a_d, pitch_a, n_a, 
				vg_b_d, pitch_b, n_b, k, d_d, pitch_d, p);
			break;
		case BINARY:
			binary_kernel_same<<<grid, block>>>(vg_a_d, pitch_a, n_a, 
				vg_b_d, pitch_b, n_b, k, d_d, pitch_d, p);
			break;
		case MINKOWSKI:
			minkowski_kernel_same<<<grid, block>>>(vg_a_d, pitch_a, 
				n_a, vg_b_d, pitch_b, n_b, k, d_d, pitch_d, p);
			break;
		default:
			fprintf(stderr, "unknown distance method");
			exit(EXIT_FAILURE);
	}
}

void distance(const float * vg_a, size_t pitch_a, size_t n_a,
	const float * vg_b, size_t pitch_b, size_t n_b,
	size_t k, float * d, size_t pitch_d, dist_method method, float p)
{
	size_t 
		pitch_a_d, pitch_b_d, pitch_d_d;
	int same = (vg_a == vg_b); // are the two sets of vectors the same?

   	// Allocate space for the vectors and distances on the gpu
	cudaMallocPitch((void**)&distance_vg_a_d, &pitch_a_d, k * sizeof(float), 
		n_a);
	cudaMemcpy2D(distance_vg_a_d, pitch_a_d, vg_a, pitch_a, k * sizeof(float), 
		n_a, cudaMemcpyHostToDevice);
	cudaMallocPitch((void**)&distance_d_d, &pitch_d_d, n_a * sizeof(float), 
		n_b);

	checkCudaError("distance function : malloc and memcpy");
    
	if(same) // don't need to move vg_b to gpu 
		distance_device(distance_vg_a_d, pitch_a_d, n_a, 
			distance_vg_a_d, pitch_a_d, n_a, k, distance_d_d, pitch_d_d,
			method, p);
	else { // vg_b is a different set of pnts so store it on gpu too
		cudaMallocPitch((void**)&distance_vg_b_d, &pitch_b_d, 
			k * sizeof(float), n_b);
		cudaMemcpy2D(distance_vg_b_d, pitch_b_d, vg_b, pitch_b, 
			k * sizeof(float), n_b, cudaMemcpyHostToDevice);

		checkCudaError("distance function : malloc and memcpy");

		distance_device(distance_vg_a_d, pitch_a_d, n_a, distance_vg_b_d, 
			pitch_b_d, n_b, k, distance_d_d, pitch_d_d, method, p);
	}
	checkCudaError("distance function : kernel invocation");
	// Copy the result back to cpu land now that gpu work is done
	cudaMemcpy2D(d, pitch_d, distance_d_d, pitch_d_d, n_a * sizeof(float), 
		n_b, cudaMemcpyDeviceToHost);
	checkCudaError("distance function : memcpy");
    
	// Free allocated space
	cudaFree(distance_vg_a_d);
	cudaFree(distance_vg_b_d);
	cudaFree(distance_d_d);
}

void distanceLeaveOnGpu(dist_method method, float p, const float * points, 
	size_t dim, size_t numPoints, 
	float ** gpuDistances, size_t * pitchDistances) // outputs
{
	size_t pitchPoints;
	float * dPoints;

   	// prepare the vectors and distance storage on the gpu
	cudaMallocPitch((void**)&dPoints, 
		&pitchPoints, dim * sizeof(float), numPoints);
	cudaMemcpy2D(dPoints, pitchPoints, points, 
		dim * sizeof(float), dim * sizeof(float), numPoints, 
		cudaMemcpyHostToDevice);
	cudaMallocPitch((void**)gpuDistances, pitchDistances, 
		numPoints * sizeof(float), numPoints);
	checkCudaError("distance on gpu func : malloc and memcpy");
    
	distance_device(dPoints, pitchPoints, numPoints, 
		dPoints, pitchPoints, numPoints, dim, *gpuDistances, *pitchDistances, 
		method, p);
	checkCudaError("distance on gpu func : kernel invocation");
        
	// clean up resources
	cudaFree(dPoints); // be kind rewind
}
