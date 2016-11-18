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

void R_init_mylib(DllInfo *info) {
  R_CallMethodDef callMethods[]  = {
    {"gpuMatMult", (DL_FUNC) &gpuMatMult, 2},
    {NULL, NULL, 0}
  };
  R_registerRoutines(info, NULL, callMethods, NULL, NULL);
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

void rformatInput(const int * images, 
                  const int * xcoords, const int * ycoords, const int * zcoords,
                  const int * mins, const int * maxes,
                  const float * evs, const int * numrows, const int * numimages, 
                  float * output)
{
  getData(images, xcoords, ycoords, zcoords, mins, maxes, evs, 
          *numrows, *numimages, output);
}

void rformatOutput(const int * imageList1, const int * numImages1, 
                   const int * imageList2, const int * numImages2, 
                   const int * structureid,
                   const double * cutCorrelation, const int * cutPairs,
                   const double * correlations, const double * signifs, const int * numPairs, 
                   double * results, int * nrows)
{
  *nrows = (int) parseResults(imageList1, *numImages1, imageList2, 
                              *numImages2, *structureid, *cutCorrelation, *cutPairs, 
                              correlations, signifs, numPairs, results);
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

void rhostT(const float * pairs, const float * coeffs, const int * n, 
            float * ts) 
{
  hostSignif(pairs, coeffs, (size_t) *n, ts);
}

void rSignifFilter(const double * data, int * rows, double * results) {
  *rows = signifFilter(data, (size_t) *rows, results);
}

void gSignifFilter(const float * data, int * rows, float * results) {
  *rows = gpuSignifFilter(data, (size_t) *rows, results);
}

void RcublasPMCC(const float * samplesA, const int * numSamplesA,
                 const float * samplesB, const int * numSamplesB,
                 const int * sampleSize,
                 float * correlations)
{
  cublasPMCC(samplesA, *numSamplesA, samplesB, *numSamplesB, *sampleSize, 
             correlations);
}

void RhostKendall(const float * X, const float * Y, const int * n, 
                  double * answer)
{
  *answer = hostKendall(X, Y, *n);
}

void RpermHostKendall(const float * X, const int * nx, const float * Y, 
                      const int * ny, const int * sampleSize, double * answers)
{
  permHostKendall(X, *nx, Y, *ny, *sampleSize, answers);
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

  presub = Calloc(len, int);
  presup = Calloc(len, int);

  hclusterPreparedDistances(gpuDistances, dpitch, *numPoints, 
                            presub, presup,
                            val,
                            hcmeth,
                            lambda, beta);

  formatClustering(len, presub, presup, merge, order);

  Free(presub);
  Free(presup);
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

  presub = Calloc(len, int);
  presup = Calloc(len, int);

  hcluster(distMat, pitch, *numPoints, presub, presup, val, nummethod,
           lambda, beta);

  formatClustering(len, presub, presup, merge, order);

  Free(presub);
  Free(presup);
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

void RgetQRDecomp(int * rows, int * cols, float * a, float * q, int * pivot,
                  int * rank)
{

  int
    fbytes = sizeof(float),
    m = *rows, n = *cols;
  float
    * da, * dq;

  cublasAlloc(m*n, fbytes, (void **)&da);
  cublasAlloc(m*m, fbytes, (void **)&dq);
  cublasSetMatrix(m, n, fbytes, a, m, da, m);

  getQRDecomp(m, n, dq, da, pivot);

  cublasGetMatrix(m, n, fbytes, da, m, a, m);
  cublasGetMatrix(m, m, fbytes, dq, m, q, m);
  cublasFree(da);
  cublasFree(dq);

  int foundZero = 0;
  for(int i = 0; (i < m) && (i < n); i++) {
    if((a[i+i*m] < 0.0001f) && (a[i+i*m] > -0.0001f)) {
      foundZero = 1;
      *rank = i+1;
      break;
    }
  }
  if(!foundZero) {
    if(m > n) *rank = n;
    else *rank = m;
  }
}

// solve for B:  XB=Y where B and Y are vectors and X is a matrix of
// dimension rows x cols
void RqrSolver(int * rows, int * cols, float * matX, float * vectY, 
               float * vectB)
{
  int
    fbytes = sizeof(float),
    m = *rows, n = *cols;
  float
    * dX, * dY, * dB;

  cublasAlloc(m*n, fbytes, (void **)&dX);
  cublasAlloc(n, fbytes, (void **)&dB);
  cublasAlloc(m, fbytes, (void **)&dY);
  checkCublasError("RqrSolver: line 80");

  cublasSetMatrix(m, n, fbytes, matX, m, dX, m);
  cublasSetVector(m, fbytes, vectY, 1, dY, 1);
  checkCublasError("RqrSolver: line 84");

  qrSolver(m, n, dX, dY, dB);

  cublasFree(dX);
  cublasFree(dY);

  cublasGetVector(n, fbytes, dB, 1, vectB, 1);
  checkCublasError("RqrSolver: line 93");

  cublasFree(dB);
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
               int *rank, int *pivot, double * qrAux, int useSingle)
{
  if (useSingle) {
    gpuLSFitF(X, *n, *p, Y, *nY, *tol, coeffs, resids, effects,
              rank, pivot, qrAux);
  }
  else {
    //    gpuLSFitD(X, *n, *p, Y, *nY, *tol, coeffs, resids, effects, rank, pivot, qrAux);
  }    
}
