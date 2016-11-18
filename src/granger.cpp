#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#include "Rmath.h"
#include "cuda_runtime_api.h"

#include "cuseful.h"
#include "cudaUtils.h"
#include "granger.h"

#define max(a, b) ((a > b)?a:b)

#define THREADSPERDIM   16

#define FALSE 0
#define TRUE !FALSE

void getPValues(int rows, int cols, const float * fStats, int p, int embedRows,
                float * pValues)
{
  float fscore = 0.f;

  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      fscore = fStats[i + j * rows];
      pValues[i + j * rows] = 1.f - (float) pf((double) fscore,
                                               (double) p, (double)embedRows - 2.0 * (double) p - 1.0,
                                               1, 0);
    }
  }
}

void granger(int rows, int cols, const float * y, int p, 
             float * fStats, float * pValues)
{
  if(cols < 2) {
    fatal("The Granger test needs at least 2 variables.\n");
    return;
  }
  int
    i, j, k, t = p+1,
    fbytes = sizeof(float),
    embedRows = rows-p, embedCols = t*2;
  float 
    * Y, * rQ, * rR,
    * unrQ, * unrR,
    * restricted, * unrestricted,
    * rdata, * unrdata,
    * dfStats; // * dpValues;
  size_t 
    size = cols*cols*fbytes, partSize = embedRows*size;

  cudaMalloc((void **)&Y, embedCols*partSize);

  cudaMalloc((void **)&rQ, t*embedRows*cols*fbytes);
  cudaMalloc((void **)&rR, t*t*cols*fbytes);
  cudaMalloc((void **)&rdata, t*embedRows*cols*fbytes);
  cudaMalloc((void **)&unrdata, (embedCols-1)*partSize);
  cudaMalloc((void **)&restricted, t*cols*fbytes);
  if( hasCudaError("granger: line 267: gpu memory allocation") ) return;

  int
    Ydim =  embedCols * embedRows,
    rQdim = t * embedRows, rRdim = t * t,
    rdataDim = t*embedRows, restrictedDim = t,
    unrQdim = (embedCols-1) * embedRows, 
    unrRdim = (embedCols-1) * (embedCols-1),
    unrestrictedDim = embedCols-1, unrdataDim = (embedCols-1)*embedRows;
  float 
    * ypos, * rdataPos, * unrdataPos,
    * evenCols;
  int 
    skip = 2*embedRows, colBytes = embedRows*fbytes;
  const float 
    * vectA, * vectB;

  for(i = 0; i < cols; i++) {
    rdataPos = rdata+i*rdataDim;
    evenCols = rdataPos+embedRows;
    vectA = y+i*rows; 
    for(j = 0; j < cols; j++) {
      if(i == j) continue;
                        
      ypos = Y+(i*cols+j)*Ydim;
      unrdataPos = unrdata+(i*cols+j)*unrdataDim;

      vectB = y+j*rows;

      for(k = 0; k < p+1; k++) { // produce t subcols
        cudaMemcpy(ypos+k*skip, vectA+(p-k), embedRows*fbytes, 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ypos+k*skip+embedRows, vectB+(p-k), 
                   embedRows*fbytes, cudaMemcpyHostToDevice);
      }
      cudaMemcpy(unrdataPos+embedRows, ypos+skip,
                 (embedCols-2)*embedRows*fbytes, cudaMemcpyDeviceToDevice);
    }
    // build restricted data from last set of unrestricted data
    // only need one per column, not one for each pairing
    for(k = 0; k < embedCols-2; k+=2) { 
      cudaMemcpy(evenCols+(k*embedRows)/2, unrdataPos+(1+k)*embedRows,
                 colBytes, cudaMemcpyDeviceToDevice);
    }
  }
  if( hasCudaError("granger : mem copy from host to device") ) return;

  int numBlocks = cols / THREADSPERDIM;
  if(numBlocks * THREADSPERDIM < cols) numBlocks++;

  dim3 
    dimRGrid(numBlocks), 
    dimRBlock(THREADSPERDIM), 
    dimUnrGrid(numBlocks, numBlocks), 
    dimUnrBlock(THREADSPERDIM, THREADSPERDIM);

  void * restArgs[] = {
    &cols, &cols,
    &embedRows,
    &t,
    &rdata, 
    &rdataDim,
    &Y,
    &Ydim,
    &rQ,
    &rQdim,
    &rR,
    &rRdim,
    &restricted,
    &restrictedDim
  };
  cudaLaunch("getRestricted", restArgs, dimRGrid, dimRBlock);
  if( hasCudaError("granger: getRestricted kernel execution") ) return;

  cudaFree(rQ);
  cudaFree(rR);

  cudaMalloc((void **)&unrQ, (embedCols-1)*partSize);
  cudaMalloc((void **)&unrR, (embedCols-1)*(embedCols-1)*size);
  cudaMalloc((void **)&unrestricted, (embedCols-1)*size);
  if( hasCudaError("granger: line 336: attemped gpu memory allocation") ) 
    return;

  size_t unrestT = embedCols - 1;
  void * unrestArgs[] = {
    &cols, &cols,
    &embedRows,
    &unrestT,
    &unrdata, 
    &unrdataDim,
    &Y,
    &Ydim,
    &unrQ,
    &unrQdim,
    &unrR,
    &unrRdim,
    &unrestricted,
    &unrestrictedDim
  };
  cudaLaunch("getUnrestricted", unrestArgs, dimUnrGrid, dimUnrBlock);
  if( hasCudaError("granger : getUnRestricted kernel execution") ) return;

  cudaFree(unrQ);
  cudaFree(unrR);

  size_t resultSize = cols*cols*fbytes;
  cudaMalloc((void **)&dfStats, resultSize);
  // cudaMalloc((void **)&dpValues, resultSize);
  if( hasCudaError("granger: line 350: gpu memory allocation") ) return;

  int diagFlag = FALSE;
  void * ftestArgs[] = {
    &diagFlag,
    &p,
    &embedRows,
    &cols, &cols,
    &t, &unrestT,
    &Y, &Ydim,
    &restricted, &restrictedDim, 
    &unrestricted, &unrestrictedDim,
    &rdata, &rdataDim,
    &unrdata, &unrdataDim,
    &dfStats
  };
  cudaLaunch("ftest", ftestArgs, dimUnrGrid, dimUnrBlock); 
  if( hasCudaError("granger : ftest kernel execution") ) return;

  cudaMemcpy(fStats, dfStats, resultSize, cudaMemcpyDeviceToHost);
  // cudaMemcpy(pValues, dpValues, resultSize, cudaMemcpyDeviceToHost);
  if( hasCudaError("granger : mem copy device to host") ) return;

  getPValues(cols, cols, fStats, p, embedRows, pValues);

  cudaFree(Y);
  cudaFree(restricted);
  cudaFree(unrestricted);
  cudaFree(rdata);
  cudaFree(unrdata);
  cudaFree(dfStats);
  // cudaFree(dpValues);
}

void grangerxy(int rows, int colsx, const float * x, int colsy, 
               const float * y, int p, float * fStats, float * pValues)
{

  if((p < 0) || (rows < 1) || (colsx < 1) || (colsy < 1)) {
    fatal("The Granger XY test needs at least a pair variables.\n");
    return;
  }
  int
    i, j, k, t = p+1,
    fbytes = sizeof(float),
    embedRows = rows-p, embedCols = t*2;
  float 
    * Y, * rQ, * rR,
    * unrQ, * unrR,
    * restricted, * unrestricted,
    * rdata, * unrdata,
    * dfStats; // * dpValues;
  size_t 
    size = colsx*colsy*fbytes, partSize = embedRows*size;

  cudaMalloc((void **)&Y, embedCols*partSize);

  cudaMalloc((void **)&rQ, t*embedRows*colsy*fbytes);
  cudaMalloc((void **)&rR, t*t*colsy*fbytes);
  cudaMalloc((void **)&rdata, t*embedRows*colsy*fbytes);
  cudaMalloc((void **)&restricted, t*colsy*fbytes);

  cudaMalloc((void **)&unrQ, (embedCols-1)*partSize);
  cudaMalloc((void **)&unrR, (embedCols-1)*(embedCols-1)*size);
  cudaMalloc((void **)&unrestricted, (embedCols-1)*size);
  cudaMalloc((void **)&unrdata, (embedCols-1)*partSize);
  checkCudaError("grangerxy : attemped gpu memory allocation");

  int
    Ydim =  embedCols * embedRows,
    rQdim = t * embedRows, rRdim = t * t,
    rdataDim = t*embedRows, restrictedDim = t,
    unrQdim = (embedCols-1) * embedRows, 
    unrRdim = (embedCols-1) * (embedCols-1),
    unrestrictedDim = embedCols-1, unrdataDim = (embedCols-1)*embedRows;
  float 
    * ypos, * rdataPos, * unrdataPos;

  int 
    skip = 2*embedRows, colBytes = embedRows*fbytes;
  const float * vectA, * vectB;
  float * evenCols;

  for(i = 0; i < colsy; i++) {
    rdataPos = rdata+i*rdataDim;
    evenCols = rdataPos+embedRows;
    vectA = y+i*rows; 
    for(j = 0; j < colsx; j++) {
      ypos = Y+(i*colsx+j)*Ydim;
      unrdataPos = unrdata+(i*colsx+j)*unrdataDim;

      vectB = x+j*rows;

      for(k = 0; k < p+1; k++) { // produce t subcols
        cudaMemcpy(ypos+k*skip, vectA+(p-k), embedRows*fbytes, 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ypos+k*skip+embedRows, vectB+(p-k), 
                   embedRows*fbytes, cudaMemcpyHostToDevice);
      }
      cudaMemcpy(unrdataPos+embedRows, ypos+skip,
                 (embedCols-2)*embedRows*fbytes, cudaMemcpyDeviceToDevice);
    }
    // build restricted data from last set of unrestricted data
    // only need one per column, not one for each pairing
    for(k = 0; k < embedCols-2; k+=2) { 
      cudaMemcpy(evenCols+(k*embedRows)/2, unrdataPos+(1+k)*embedRows, 
                 colBytes, cudaMemcpyDeviceToDevice);
    }
    char errline[16];
    sprintf(errline, "gxy err : %d\n", i);
    if( hasCudaError(errline) ) return;
  }
  checkCudaError("grangerxy : mem copy from host to device");

  int 
    numBlocksX = colsx / THREADSPERDIM,
    numBlocksY = colsy / THREADSPERDIM;

  if(numBlocksX * THREADSPERDIM < colsx) numBlocksX++;
  if(numBlocksY * THREADSPERDIM < colsy) numBlocksY++;

  dim3 
    dimRGrid(numBlocksY), 
    dimRBlock(THREADSPERDIM), 
    dimUnrGrid(numBlocksX, numBlocksY), 
    dimUnrBlock(THREADSPERDIM, THREADSPERDIM);

  void * restArgs[] = {
    &colsx, &colsy,
    &embedRows,
    &t,
    &rdata, &rdataDim,
    &Y, &Ydim,
    &rQ, &rQdim,
    &rR, &rRdim,
    &restricted, &restrictedDim
  };
  cudaLaunch("getRestricted", restArgs,
                    dimRGrid, dimRBlock);
  
  size_t unrestT = embedCols - 1;
  void * unrestArgs[] = {
    &colsx, &colsy,
    &embedRows,
    &unrestT,
    &unrdata, &unrdataDim,
    &Y, &Ydim,
    &unrQ, &unrQdim,
    &unrR, &unrRdim,
    &unrestricted, &unrestrictedDim
  };
  cudaLaunch("getUnrestricted", unrestArgs,
                    dimUnrGrid, dimUnrBlock);

  checkCudaError("grangerxy : kernel execution get(Un)Restricted");

  cudaFree(rQ);
  cudaFree(unrQ);
  cudaFree(rR);
  cudaFree(unrR);

  size_t resultSize = colsx*colsy*fbytes;
  cudaMalloc((void **)&dfStats, resultSize);
  // cudaMalloc((void **)&dpValues, resultSize);
  checkCudaError("grangerxy : attemped gpu memory allocation");

  int diagFlag = TRUE;
  void * ftestArgs[] = {
    &diagFlag,
    &p,
    &embedRows,
    &colsx, &colsy,
    &t, &unrestT,
    &Y, &Ydim,
    &restricted, &restrictedDim, 
    &unrestricted, &unrestrictedDim,
    &rdata, &rdataDim,
    &unrdata, &unrdataDim,
    &dfStats
  };
  cudaLaunch("ftest", ftestArgs, dimUnrGrid, dimUnrBlock); 
  checkCudaError("grangerxy : kernel execution ftest");

  cudaMemcpy(fStats, dfStats, resultSize, cudaMemcpyDeviceToHost);
  // cudaMemcpy(pValues, dpValues, resultSize, cudaMemcpyDeviceToHost);
  checkCudaError("grangerxy : mem copy from device to host");

  getPValues(colsx, colsy, fStats, p, embedRows, pValues);

  cudaFree(Y);
  cudaFree(restricted);
  cudaFree(unrestricted);
  cudaFree(rdata);
  cudaFree(unrdata);
  cudaFree(dfStats);
  // cudaFree(dpValues);
}
