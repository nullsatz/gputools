#include<stdio.h>
#include<string.h>

#include "cublas.h"

#include "R.h"
#include "Rinternals.h"
#include  "R_ext/Rdynload.h"

#include "correlation.h"
#include "cuseful.h"
#include "distance.h"
#include "granger.h"
#include "hcluster.h"
#include "kendall.h"
#include "lsfit.h"
#include "matmult.h"
#include "mi.h"
#include "qrdecomp.h"

#include "rinterface.h"

extern "C" {
  SEXP gputoolsSetKernelDir(SEXP dir);
  SEXP gputoolsUnloadKernels();
}

#define CDEF(name, n) {#name, (DL_FUNC) &name, n, NULL}
#define CALLDEF(name, n) {#name, (DL_FUNC) &name, n}

static const R_CMethodDef cMethods[] = {
  CDEF(rpmcc, 9),
  CDEF(rsetDevice, 1),
  CDEF(rgetDevice, 1),
  CDEF(rtestT, 4),
  CDEF(RgpuKendall, 6),
  CDEF(rgpuGranger, 6),
  CDEF(rgpuGrangerXY, 8),
  CDEF(Rdistclust, 8),
  CDEF(Rdistances, 6),
  CDEF(Rhcluster, 6),
  CDEF(rGetQRDecompRR, 7),
  CDEF(rGetInverseFromQR, 5),
  CDEF(rSolveFromQR, 6),
  CDEF(rBSplineMutualInfo, 8),
  CDEF(RgpuLSFit, 13),
  {NULL, NULL, 0, NULL}
};

static const R_CallMethodDef callMethods[] = {
  CALLDEF(gpuMatMult, 2),
  CALLDEF(gputoolsSetKernelDir, 1),
  CALLDEF(gputoolsUnloadKernels, 0),
  {NULL, NULL, 0}
};

extern "C" void R_init_gputools(DllInfo * dll)
{
  R_registerRoutines(dll, cMethods, callMethods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}

// whichObs = 0 means everything
// whichObs = 1 means pairwiseComplete
void rpmcc(const int * whichObs,
           const float * samplesA, const int * numSamplesA,
           const float * samplesB, const int * numSamplesB,
           const int * sampleSize,
           float * numPairs, float * correlations, float * signifs)
{
  UseObs myObs;
  switch(*whichObs) {
  case 0:
    myObs = everything;
    break;
  case 1:
    myObs = pairwiseComplete;
    break;
  default:
    fatal("unknown use method");
  }
  pmcc(myObs,
       samplesA, *numSamplesA,
       samplesB, *numSamplesB,
       *sampleSize,
       numPairs, correlations, signifs);
}

void rsetDevice(const int * device) {
  setDevice(*device);
}

void rgetDevice(int * device) {
  getDevice(device);
}

void rtestT(const float * pairs, const float * coeffs, const int * n, 
            float * ts)
{
  testSignif(pairs, coeffs, (size_t) *n, ts);
}

void RgpuKendall(const float * X, const int * nx, const float * Y, 
                 const int * ny, const int * sampleSize, double * answers)
{
  masterKendall(X, *nx, Y, *ny, *sampleSize, answers);
}

void rgpuGranger(const int * rows, const int * cols, const float * y,
                 const int * p, float * fStats, float * pValues)
{
  granger(*rows, *cols, y, *p, fStats, pValues);
}

void rgpuGrangerXY(const int * rows, const int * colsx, const float * x, 
                   const int * colsy, const float * y, const int * p, 
                   float * fStats, float * pValues)
{
  grangerxy(*rows, *colsx, x, *colsy, y, *p, fStats, pValues);
}

dist_method getDistEnum(const char * methodStr)
{
  if(0 == strcmp(methodStr,"maximum"))    return MAXIMUM;
  if(0 == strcmp(methodStr,"manhattan"))  return MANHATTAN;
  if(0 == strcmp(methodStr,"canberra"))   return CANBERRA;
  if(0 == strcmp(methodStr,"binary"))     return BINARY;
  if(0 == strcmp(methodStr,"minkowski"))  return MINKOWSKI;
  //      if(0 == strcmp(methodStr,"dot"))                return DOT;
  return EUCLIDEAN;
}

hc_method getClusterEnum(const char * methodStr)
{
  if(0 == strcmp(methodStr,"complete"))           return COMPLETE;
  if(0 == strcmp(methodStr,"wpgma"))                      return WPGMA;
  if(0 == strcmp(methodStr,"average"))            return AVERAGE;
  if(0 == strcmp(methodStr,"median"))                     return MEDIAN;
  if(0 == strcmp(methodStr,"centroid"))           return CENTROID;
  if(0 == strcmp(methodStr,"flexible_group"))     return FLEXIBLE_GROUP;
  if(0 == strcmp(methodStr,"flexible"))           return FLEXIBLE;
  if(0 == strcmp(methodStr,"ward"))                       return WARD;
  if(0 == strcmp(methodStr,"mcquitty"))           return MCQUITTY;
  return SINGLE;
}

void Rdistclust(const char ** distmethod, const char ** clustmethod, 
                const float * points, const int * numPoints, const int * dim,
                int * merge, int * order, float * val)
{
  dist_method dmeth = getDistEnum(*distmethod); 
  hc_method hcmeth = getClusterEnum(*clustmethod); 

  size_t dpitch = 0;
  float * gpuDistances = NULL;

  distanceLeaveOnGpu(dmeth, 2.f, points, *dim, *numPoints, 
                     &gpuDistances, &dpitch);

  size_t len = (*numPoints) - 1;
  float 
    lambda = 0.5f, beta = 0.5f;
  int 
    * presub, * presup;

  presub = R_Calloc(len, int);
  presup = R_Calloc(len, int);

  hclusterPreparedDistances(gpuDistances, dpitch, *numPoints, 
                            presub, presup,
                            val,
                            hcmeth,
                            lambda, beta);

  formatClustering(len, presub, presup, merge, order);

  R_Free(presub);
  R_Free(presup);
}

void Rdistances(const float * points, const int * numPoints, const int * dim,
                float * distances, const char ** method, const float *p)
{
  dist_method nummethod = getDistEnum(*method); 

  distance(points, (*dim)*sizeof(float), *numPoints, points, 
           (*dim)*sizeof(float), *numPoints, *dim, distances, 
           (*numPoints)*sizeof(float), nummethod, *p);
}

void Rhcluster(const float * distMat, const int * numPoints, 
               int * merge, int * order, float * val,
               const char ** method)
{
  hc_method nummethod = getClusterEnum(*method); 
        
  size_t len = (*numPoints) - 1;
  size_t pitch = (*numPoints) * sizeof(float);
  float lambda = 0.5;
  float beta = 0.5;
  int 
    * presub, * presup;

  presub = R_Calloc(len, int);
  presup = R_Calloc(len, int);

  hcluster(distMat, pitch, *numPoints, presub, presup, val, nummethod,
           lambda, beta);

  formatClustering(len, presub, presup, merge, order);

  R_Free(presub);
  R_Free(presup);
}

void formatClustering(const int len, const int * sub,  const int * sup, 
                      int * merge, int * order)
{
  for(size_t i = 0; i < len; i++) {
    merge[i] = -(sub[i] + 1);
    merge[i+len] = -(sup[i] + 1);
  }

  for(size_t i = 0; i < len; i++) {
    for(size_t j = i+1; j < len; j++) {
      if((merge[j] == merge[i]) || (merge[j] == merge[i+len]))
        merge[j] = i + 1;
      if((merge[j+len] == merge[i]) || (merge[j+len] == merge[i+len]))
        merge[j+len] = i + 1;
      if(((merge[j+len] < 0) && (merge[j] > 0)) 
         || ((merge[j] > 0) && (merge[j+len] > 0) 
             && (merge[j] > merge[j+len]))) {
        int holder = merge[j];
        merge[j] = merge[j+len];
        merge[j+len] = holder; 
      }
    }
  }
  getPrintOrder(len, merge, order);
}

void getPrintOrder(const int len, const int * merge, int * order)
{
  int 
    level = len-1, otop = len;

  depthFirst(len, merge, level, &otop, order);
}

void depthFirst(const int len, const int * merge, int level, int * otop, 
                int * order)
{
  int
    left = level, right = level + len;

  if(merge[right] < 0) {
    order[*otop] = -merge[right];
    (*otop)--;
  } else
    depthFirst(len, merge, merge[right]-1, otop, order);

  if(merge[left] < 0) {
    order[*otop] = -merge[left];
    (*otop)--;
  } else
    depthFirst(len, merge, merge[left]-1, otop, order);
}

void rGetQRDecompRR(const int * rows, const int * cols,
                    const double * tol, float * x, int * pivot,
                    double * qraux, int * rank)
{
  float * dQR;
  cudaMalloc((void **) &dQR, (*rows) * (*cols) * sizeof(float));
  checkCudaError("rGetQRDecompRR:");

  cudaMemcpy(dQR, x, (*rows) * (*cols) * sizeof(float),
             cudaMemcpyHostToDevice);

  getQRDecompRR(*rows, *cols, *tol, dQR, pivot, qraux, rank);

  cudaMemcpy(x, dQR, (*rows) * (*cols) * sizeof(float),
             cudaMemcpyDeviceToHost);
  checkCudaError("rGetQRDecompRR:");
  cudaFree(dQR);
  checkCudaError("rGetQRDecompRR:");
}

void rGetInverseFromQR(const int * rows, const int * cols,
                       const float * q, const float * r,
                       float * inverse)
{
  float
    * dQ, * dR, * dInverse;

  cudaMalloc((void **) &dQ, (*rows) * (*cols) * sizeof(float));
  cudaMalloc((void **) &dR, (*cols) * (*cols) * sizeof(float));
  cudaMalloc((void **) &dInverse,  (*rows) * (*cols) * sizeof(float));
  checkCudaError("rGetInverseFromQR:");

  cudaMemcpy(dQ, q, (*rows) * (*cols) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dR, r, (*cols) * (*cols) * sizeof(float),
             cudaMemcpyHostToDevice);

  getInverseFromQR(*rows, *cols, dQ, dR, dInverse);

  cudaFree(dQ);
  cudaFree(dR);

  cudaMemcpy(inverse, dInverse, (*rows) * (*cols) * sizeof(float),
             cudaMemcpyDeviceToHost);
  checkCudaError("rGetInverseFromQR:");

  cudaFree(dInverse);
}

void rSolveFromQR(const int * rows, const int * cols, const float * q, const float * r,
                  const float * y, float * b)
{
  solveFromQR(*rows, *cols, q, r, y, b);
}

void rBSplineMutualInfo(int * nBins, int * splineOrder, int * nsamples,
                        int * rowsA, const float * A,
                        int * rowsB, const float * B, 
                        float * mutualInfo)
{
  bSplineMutualInfo(*nBins, *splineOrder, *nsamples, *rowsA, A, *rowsB, B,
                    mutualInfo);
}

// Interface for R functions requiring least-squares computations.
//
void RgpuLSFit(float *X, int *n, int *p, float *Y, int *nY,
               double *tol, float *coeffs, float *resids, float *effects,
               int *rank, int *pivot, double * qrAux, int * useSingle)
{
  if (*useSingle) {
    gpuLSFitF(X, *n, *p, Y, *nY, *tol, coeffs, resids, effects,
              rank, pivot, qrAux);
  }
  else {
    //    gpuLSFitD(X, *n, *p, Y, *nY, *tol, coeffs, resids, effects, rank, pivot, qrAux);
  }    
}
