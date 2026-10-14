#define NTHREADS 512

__global__ void getColNorms(int rows, int cols, real * da, int lda, 
                            real * colNorms)
{
  int colIndex = threadIdx.x + blockIdx.x * blockDim.x;
  real 
    sum = real(0), term,
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

