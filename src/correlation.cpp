#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "cuda_runtime.h"
#include<cuseful.h>
#include<R.h>
#include<Rinternals.h>
#include<correlation.h>
#include "cudaUtils.h"

#define FALSE 0
#define TRUE !FALSE

#define NUMTHREADS 16
#define THREADWORK 32

void testSignif(const float * goodPairs, const float * coeffs, 
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

  void * args[] = {
    &gpuPairs,
    &gpuCoeffs,
    &n,
    &gpuTs
  };
  cudaLaunch("gpuSignif", args, tgrid, tblock);

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

void pmcc(UseObs whichObs,
    const float * vectsa, size_t na,
    const float * vectsb, size_t nb,
    size_t dim, float * numPairs, float * correlations,
          float * signifs)
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

  void * argsMeans[] = {
    &gpuVA, &na,
    &gpuVB, &nb,
    &dim,
    &gpumean, &gpuNumPairs
  };

  void * argsSD[] = {
    &gpuVA, &na,
    &gpuVB, &nb,
    &dim,
    &gpumean,
    &gpuNumPairs,
    &gpuSds
  };

  void * argsPMCC[] = {
    &gpuVA, &na,
    &gpuVB, &nb,
    &dim,
    &gpuNumPairs,
    &gpumean,
    &gpuSds,
    &gpuCorrelations
  };

  switch(whichObs) {
  case pairwiseComplete:
    cudaLaunch("gpuMeans", argsMeans, grid, block);

    cudaLaunch("gpuSD", argsSD, grid, block);

    cudaLaunch("gpuPMCC", argsPMCC, grid, block);
    break;
  default:
    cudaLaunch("gpuMeansNoTest", argsMeans, grid, block);

    cudaLaunch("gpuSDNoTest", argsSD, grid, block);
    
    cudaLaunch("gpuPMCCNoTest", argsPMCC, grid, block);
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

