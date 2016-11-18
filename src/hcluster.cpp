#include <stdio.h>
#include <string.h>
#include <string>

#include "R.h"

#include "math_constants.h"
#include "cuseful.h"
#include "hcluster.h"
#include "cudaUtils.h"

#define NUM_THREADS 32
#define NUM_BLOCKS 1024

void hcluster(const float * dist, size_t dist_pitch, size_t n,
              int * sub, int * sup, float * val, hc_method method,
              float lambda, float beta)
{
  // Allocate space for the distance matrix
  size_t pitch_dist_d;
  float * hcluster_dist_d; // Distance matrix
  cudaMallocPitch((void**)&hcluster_dist_d, &pitch_dist_d,
                  n * sizeof(float), n);

  // Copy the distance matrix
  cudaMemcpy2D(hcluster_dist_d, pitch_dist_d,
               dist, dist_pitch, 
               n * sizeof(float), n,
               cudaMemcpyHostToDevice);
  
  hclusterPreparedDistances(hcluster_dist_d, pitch_dist_d, n,
                            sub, sup,
                            val,
                            method, lambda, beta);

  cudaFree(hcluster_dist_d);
  checkCudaError("hcluster : cudaFree");
}

void hclusterPreparedDistances(float * gpuDist, size_t pitch_dist_d, size_t n,
                               int * sub, int * sup,
                               float * val,
                               hc_method method,
                               float lambda, float beta)
{
  float
    * hcluster_count_d, // Number of elements in each cluster
    * hcluster_min_val_d, // find min of each row and
    * hcluster_merge_val_d; // Array of the values merged at

  // col containing the min of each row
  size_t * hcluster_min_col_d;

  // Arrays telling which cluster merged with which cluster
  int
    * hcluster_sub_d,
    * hcluster_sup_d;

  cudaMalloc((void**)&hcluster_count_d, n * sizeof(float));
  cudaMalloc((void**)&hcluster_min_val_d, n * sizeof(float));
  cudaMalloc((void**)&hcluster_min_col_d, n * sizeof(size_t));
  cudaMalloc((void**)&hcluster_sub_d, (n - 1) * sizeof(int));
  cudaMalloc((void**)&hcluster_sup_d, (n - 1) * sizeof(int));
  cudaMalloc((void**)&hcluster_merge_val_d, (n - 1) * sizeof(float));

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

  size_t gpuPitch = pitch_dist_d / sizeof(float);
  void * convertArgs[] = {
    &gpuDist,
    &gpuPitch,
    &n
  };

  // Convert 0 on the diagonal to infinity
  cudaLaunch("convert_kernel", convertArgs, grid1, block1);
  checkCudaError("hcluster : convert kernel");

  std::string func;
  switch(method) {
  case COMPLETE:
    func = "complete_kernel";
    break;
  case WPGMA:
    func = "wpgma_kernel";
    break;
  case AVERAGE:
    func = "average_kernel";
    break;
  case MEDIAN:
    func = "median_kernel";
    break;
  case CENTROID:
    func = "centroid_kernel";
    break;
  case FLEXIBLE_GROUP:
    func = "flexible_group_kernel";
    break;
  case FLEXIBLE:
    func = "flexible_kernel";
    break;
  case WARD:
    func = "ward_kernel";
    break;
  case MCQUITTY:
    func = "mcquitty_kernel";
    break;
  case SINGLE:
  default:
    func = "single_kernel";
    break;
  }

  void * findMin1Args[] = {
    &gpuDist, 
    &gpuPitch,
    &n,
    &hcluster_count_d, 
    &hcluster_min_val_d,
    &hcluster_min_col_d,
    0  // place holder for row_offset (a loop variable)
  };
  void * findMin2Args[] = {
    &hcluster_min_val_d, 
    &hcluster_min_col_d,
    &hcluster_count_d, 
    &hcluster_sub_d, 
    &hcluster_sup_d, 
    &hcluster_merge_val_d,
    &n,
    0 // place holder for iter
  };
  void * funcArgs[] = {
    &gpuDist, 
    &gpuPitch,
    &n,
    &hcluster_sub_d, 
    &hcluster_sup_d,
    &hcluster_count_d,
    &hcluster_merge_val_d,
    0,  // place holder for iter
    0, // place holder for col_offset
    &lambda,
    &beta
  };
  size_t skip = NUM_BLOCKS * NUM_THREADS;
  // Merge items n - 1 times
  for(size_t iter = 0; iter < (n - 1); ++iter) {
    // Find the minimum of each column
    for(size_t row_offset = 0; row_offset < n; row_offset += NUM_BLOCKS) {
      findMin1Args[6] = &row_offset;
      cudaLaunch("find_min1_kernel", findMin1Args,
                        grid0, block0);
    }
                
    // Find overall winner; update arrays sub, sup, val, count
    findMin2Args[7] = &iter;
    cudaLaunch("find_min2_kernel", findMin2Args,
                      grid1, block1);

    // Update the distance matrix
    funcArgs[7] = &iter;
    for(size_t col_offset = 0; col_offset < n; col_offset += skip) {
      funcArgs[8] = &col_offset;
      cudaLaunch(func, funcArgs, grid0, block0);
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

  cudaFree(hcluster_count_d);
  cudaFree(hcluster_min_val_d);
  cudaFree(hcluster_min_col_d);
  cudaFree(hcluster_sub_d);
  cudaFree(hcluster_sup_d);
  cudaFree(hcluster_merge_val_d);

  checkCudaError("hcluster : cudaFree");
}
