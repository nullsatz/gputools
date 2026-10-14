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
