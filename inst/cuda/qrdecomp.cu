#define NTHREADS 512

__global__ void getColNorms(int rows, int cols, float * da, int lda, 
                            float * colNorms)
{
  int colIndex = threadIdx.x + blockIdx.x * blockDim.x;
  float 
    sum = 0.f, term,
    * col;

  if(colIndex >= cols)
    return;

  col = da + colIndex * lda;

  // debug printing
  // printf("printing column %d\n", colIndex);
  // for(int i = 0; i < rows; i++)
  // printf("%f, ", col[i]);
  // puts("");
  // end debug printing

  for(int i = 0; i < rows; i++) {
    term = col[i];
    term *= term;
    sum += term;
  }

  // debug printing
  // printf("norm %f\n", norm);
  // end debug printing

  colNorms[colIndex] = sum;
}

__global__ void gpuFindMax(int n, float * data, int threadWorkLoad, 
                           int * maxIndex)
{
  int
    j, k,
    start = threadWorkLoad * threadIdx.x,
    end = start + threadWorkLoad;
  __shared__ int maxIndicies[NTHREADS];

  maxIndicies[threadIdx.x] = -1;

  if(start >= n)
    return;

  int localMaxIndex = start;
  for(int i = start+1; i < end; i++) {
    if(i >= n)
      break;
    if(data[i] > data[localMaxIndex])
      localMaxIndex = i;
  }
  maxIndicies[threadIdx.x] = localMaxIndex;
  __syncthreads();

  for(int i = blockDim.x >> 1; i > 0; i >>= 1) {
    if(threadIdx.x < i) {
      j = maxIndicies[threadIdx.x];
      k = maxIndicies[i + threadIdx.x];
      if((j != -1) && (k != -1) && (data[j] < data[k])) 
        maxIndicies[threadIdx.x] = k;
    }
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    *maxIndex = maxIndicies[0];
    // debug printing
    // printf("max index: %d\n", *maxIndex);
    // printf("max norm: %f\n", data[*maxIndex]);
    // end debug printing
  }
}

__global__ void gpuSwapCol(int rows, float * dArray, int coli, int * dColj,
                           int * dPivot)
{
  int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if(rowIndex >= rows)
    return;

  int colj = coli + (*dColj);
  float fholder;

  fholder = dArray[rowIndex+coli*rows];
  dArray[rowIndex+coli*rows] = dArray[rowIndex+colj*rows];
  dArray[rowIndex+colj*rows] = fholder;

  if((blockIdx.x == 0) && (threadIdx.x == 0)) {
    int iholder = dPivot[coli];
    dPivot[coli] = dPivot[colj];
    dPivot[colj] = iholder;
  }
}

__global__ void makeHVector(int rows, float * input, float * output)
{
  int
    i, j;
  float 
    elt, sum;
  __shared__ float 
    beta, sums[NTHREADS];

  if(threadIdx.x >= rows)
    return;

  sum = 0.f;
  for(i = threadIdx.x ; i < rows; i += NTHREADS) {
    if((threadIdx.x == 0) && (i == 0))
      continue;
    elt = input[i];
    output[i] = elt;
    sum += elt * elt;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
        
  for(i = blockDim.x >> 1; i > 0 ; i >>= 1) {
    j = i+threadIdx.x;
    if((threadIdx.x < i) && (j < rows)) 
      sums[threadIdx.x] += sums[j];
    __syncthreads();
  }

  if(threadIdx.x == 0) {
    elt = input[0];
    float norm = sqrtf(elt * elt + sums[0]);

    if(elt > 0) 
      elt += norm;
    else 
      elt -= norm;

    output[0] = elt;

    norm = elt * elt + sums[0];
    beta = sqrtf(2.f / norm);
  }
  __syncthreads();

  for(i = threadIdx.x; i < rows; i += NTHREADS)
    output[i] *= beta;
}

// Updates the column norms by subtracting the Hadamard-square of the
// Householder vector.
//
// N.B.:  Overflow incurred in computing the square should already have
// been detected in the original norm construction.

__global__ void UpdateHHNorms(int cols, float *dV, float *dNorms) {
  // Copyright 2009, Mark Seligman at Rapid Biologics, LLC.  All rights
  // reserved.

  int colIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (colIndex < cols) {
    float val = dV[colIndex];
    dNorms[colIndex] -= val * val;
  }
}
