#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<R.h>
#include<cuseful.h>
#include<mi.h>

#include "cudaUtils.h"

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


void bSplineMutualInfo(int nbins, int order, int nsamples,
                       int nx, const float * x,
                       int ny, const float * y,
                       float * out_mi)
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
    dim3
      grid(nblocks[i]),
      block(NTHREADS);

    size_t xpitch = pitch[i] / sizeof(float);

    void * scaleArgs[] = {
      &knot_max,
      size + i,
      &nsamples,
      dx + i,
      &xpitch
    };
    cudaLaunch("scale", scaleArgs,
                      grid, block, stream[i]);

    size_t pitch_bins_i = pitch_bins[i] / sizeof(float);
    void * gbsArgs[] = {
      &nbins, &order,
      &nknots, &dknots,
      &nsamples, size + i,
      dx + i, &xpitch,
      dbins + i,
      &pitch_bins_i
    };
    cudaLaunch("get_bin_scores", gbsArgs,
                      grid, block, stream[i]);

    void * entropyArgs[] = {
      &nbins,
      &nsamples,
      size + i,
      dbins + i,
      &pitch_bins_i,
      dentropy + i
    };
    cudaLaunch("get_entropy", entropyArgs,
                      grid, block, stream[i]);
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

  pitch_bins[0] /= sizeof(float);
  pitch_bins[1] /= sizeof(float);

  size_t dpitch_mi = pitch_mi / sizeof(float);
  
  void * miArgs[] = {
    &nbins, &nsamples,
    &nx, dbins, pitch_bins, dentropy,
    &ny, dbins + 1, pitch_bins + 1, dentropy + 1,
    &dmi, &dpitch_mi
  };
  cudaLaunch("get_mi", miArgs, gridDim, blockDim);
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
