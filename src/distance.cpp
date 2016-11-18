#include <stdio.h>
#include <string.h>
#include <string>

#include "R.h"
#include "Rinternals.h"

#include "cuda_runtime_api.h"

#include "cuseful.h"
#include "distance.h"

#include "cudaUtils.h"

#define NUM_THREADS 32

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
    error("unknown distance method");
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
                     size_t k,
                     float * d_d, size_t pitch_d,
                     dist_method method, float p)
{
  dim3 block(NUM_THREADS, 1, 1);
  dim3 grid(n_a, n_b, 1);

  size_t fbytes = sizeof(float);

  pitch_a /= fbytes;
  pitch_b /= fbytes;
  pitch_d /= fbytes;

  void * args[] = {
    &vg_a_d, &pitch_a, &n_a,
    &vg_b_d, &pitch_b, &n_b,
    &k,
    &d_d, &pitch_d, &p
  };

  std::string kernelName;
  
  switch(method) {  // Calculate the distance
  case EUCLIDEAN:
    kernelName = "euclidean_kernel_same";
    break;
  case MAXIMUM:
    kernelName = "maximum_kernel_same";
    break;
  case MANHATTAN:
    kernelName = "manhattan_kernel_same";
    break;
  case CANBERRA:
    kernelName = "canberra_kernel_same";
    break;
  case BINARY:
    kernelName = "binary_kernel_same";
    break;
  case MINKOWSKI:
    kernelName = "minkowski_kernel_same";
    break;
  default:
    kernelName = "";
    error("unknown distance method");
  }
  cudaLaunch(kernelName, args, grid, block);
}

void distance(const float * vg_a, size_t pitch_a, size_t n_a,
              const float * vg_b, size_t pitch_b, size_t n_b,
              size_t k,
              float * d, size_t pitch_d,
              dist_method method, float p)
{
  size_t 
    pitch_a_d, pitch_b_d, pitch_d_d;
  int same = (vg_a == vg_b); // are the two sets of vectors the same?

  // Space for the vector data
  float * distance_vg_a_d;
  float * distance_vg_b_d;

  // Space for the resulting distance
  float * distance_d_d;

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
                    distance_vg_a_d, pitch_a_d, n_a,
                    k,
                    distance_d_d, pitch_d_d,
                    method, p);
  else { // vg_b is a different set of pnts so store it on gpu too
    cudaMallocPitch((void**)&distance_vg_b_d, &pitch_b_d, 
                    k * sizeof(float), n_b);
    cudaMemcpy2D(distance_vg_b_d, pitch_b_d, vg_b, pitch_b, 
                 k * sizeof(float), n_b, cudaMemcpyHostToDevice);

    checkCudaError("distance function : malloc and memcpy");

    distance_device(distance_vg_a_d, pitch_a_d, n_a,
                    distance_vg_b_d, pitch_b_d, n_b,
                    k,
                    distance_d_d, pitch_d_d,
                    method, p);
    cudaFree(distance_vg_b_d);
  }
  checkCudaError("distance function : kernel invocation");
  // Copy the result back to cpu land now that gpu work is done
  cudaMemcpy2D(d, pitch_d, distance_d_d, pitch_d_d, n_a * sizeof(float), 
               n_b, cudaMemcpyDeviceToHost);
  checkCudaError("distance function : memcpy");
    
  // Free allocated space
  cudaFree(distance_vg_a_d);
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
                  dPoints, pitchPoints, numPoints,
                  dim,
                  *gpuDistances, *pitchDistances, 
                  method, p);
  checkCudaError("distance on gpu func : kernel invocation");
        
  // clean up resources
  cudaFree(dPoints); // be kind rewind
}
