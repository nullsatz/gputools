#define max(a, b) ((a > b)?a:b)

#define THREADSPERDIM   16

#define FALSE 0
#define TRUE !FALSE

// mX has order rows x cols
// vectY has length rows
__global__ void getRestricted(int countx, int county, int rows, int cols, 
                              real * mX, int mXdim, real * vY, int vYdim, real * mQ, int mQdim,
                              real * mR, int mRdim, real * vectB, int vectBdim) {

  int 
    m = blockIdx.x * THREADSPERDIM + threadIdx.x, n,
    i, j, k;
  real 
    sum, invnorm,
    * X, * Y, * Q, * R, * B,
    * coli, * colj, 
    * colQ, * colX;

  if(m >= county) return;
  // Any pair (m, n) holds column m's own lags; the single-matrix form
  // leaves the diagonal pair (m, m) unfilled, so avoid n == m there, and the
  // x/y form with one x column only has n == 0.
  if(m == 1 || countx == 1) n = 0;
  else n = 1;

  X = mX + (m * mXdim);
  // initialize the intercepts
  for(i = 0; i < rows; i++)
    X[i] = real(1);

  Y = vY + (m * countx + n) * vYdim;
  B = vectB + m * vectBdim;
  Q = mQ + m * mQdim;
  R = mR + m * mRdim;

  // initialize Q with X ...
  for(i = 0; i < rows; i++) {
    for(j = 0; j < cols; j++)
      Q[i+j*rows] = X[i+j*rows];
  }

  // gramm-schmidt process to find Q
  for(j = 0; j < cols; j++) {
    colj = Q+rows*j;
    for(i = 0; i < j; i++) {
      coli = Q+rows*i;
      sum = real(0);
      for(k = 0; k < rows; k++)
        sum += coli[k] * colj[k];
      for(k = 0; k < rows; k++)
        colj[k] -= sum * coli[k];
    }
    sum = real(0);
    for(i = 0; i < rows; i++)
      sum += colj[i] * colj[i];
    invnorm = real(1) / sqrt(sum);
    for(i = 0; i < rows; i++)
      colj[i] *= invnorm;
  }
  for(i = cols-1; i > -1; i--) {
    colQ = Q+i*rows;
    // matmult Q * X -> R
    for(j = 0; j < cols; j++) {
      colX = X+j*rows;
      sum = real(0);
      for(k = 0; k < rows; k++)
        sum += colQ[k] * colX[k];
      R[i+j*cols] = sum;
    }
    sum = real(0);
    // compute the vector Q^t * Y -> B
    for(j = 0; j < rows; j++) 
      sum += colQ[j] * Y[j];
    // back substitution to find the x for Rx = B
    for(j = cols-1; j > i; j--)
      sum -= R[i+j*cols] * B[j];

    B[i] = sum / R[i+i*cols];
  }
}

// mX has order rows x cols
// vectY has length rows
__global__ void getUnrestricted(int countx, int county, int rows, int cols, 
                                real * mX, int mXdim, real * vY, int vYdim, real * mQ, int mQdim,
                                real * mR, int mRdim, real * vectB, int vectBdim) {

  int 
    n = blockIdx.x * THREADSPERDIM + threadIdx.x, 
    m = blockIdx.y * THREADSPERDIM + threadIdx.y, 
    i, j, k;
  real 
    sum, invnorm,
    * X, * Y, * Q, * R, * B,
    * coli, * colj, 
    * colQ, * colX;
  if((m >= county) || (n >= countx)) return;

  X = mX + (m * countx + n) * mXdim;
  // initialize the intercepts
  for(i = 0; i < rows; i++) 
    X[i] = real(1);

  Y = vY + (m*countx+n) * vYdim;
  B = vectB + (m*countx+n) * vectBdim;
  Q = mQ + (m*countx+n) * mQdim;
  R = mR + (m*countx+n) * mRdim;

  // initialize Q with X ...
  for(i = 0; i < rows; i++) {
    for(j = 0; j < cols; j++)
      Q[i+j*rows] = X[i+j*rows];
  }

  // gramm-schmidt process to find Q
  for(j = 0; j < cols; j++) {
    colj = Q+rows*j;
    for(i = 0; i < j; i++) {
      coli = Q+rows*i;
      sum = real(0);
      for(k = 0; k < rows; k++)
        sum += coli[k] * colj[k];
      for(k = 0; k < rows; k++)
        colj[k] -= sum * coli[k];
    }
    sum = real(0);
    for(i = 0; i < rows; i++)
      sum += colj[i] * colj[i];
    invnorm = real(1) / sqrt(sum);
    for(i = 0; i < rows; i++)
      colj[i] *= invnorm;
  }
  for(i = cols-1; i > -1; i--) {
    colQ = Q+i*rows;
    // matmult Q * X -> R
    for(j = 0; j < cols; j++) {
      colX = X+j*rows;
      sum = real(0);
      for(k = 0; k < rows; k++)
        sum += colQ[k] * colX[k];
      R[i+j*cols] = sum;
    }
    sum = real(0);
    // compute the vector Q^t * Y -> B
    for(j = 0; j < rows; j++) 
      sum += colQ[j] * Y[j];
    // back substitution to find the x for Rx = B
    for(j = cols-1; j > i; j--)
      sum -= R[i+j*cols] * B[j];

    B[i] = sum / R[i+i*cols];
  }
}

__global__ void ftest(int diagFlag, int p, int rows, int colsx, int colsy, 
                      int rCols, int unrCols, real * obs, int obsDim, 
                      real * rCoeffs, int rCoeffsDim, real * unrCoeffs, int unrCoeffsDim, 
                      real * rdata, int rdataDim, real * unrdata, int unrdataDim, 
                      real * dfStats) // real * dpValues)
{
  int 
    j = blockIdx.x * THREADSPERDIM + threadIdx.x, 
    i = blockIdx.y * THREADSPERDIM + threadIdx.y, 
    idx = i*colsx + j, k, m;
  real 
    kobs, fp = (real) p, frows = (real) rows,
    rSsq, unrSsq,
    rEst, unrEst,
    score = real(0), 
    * tObs, * tRCoeffs, * tUnrCoeffs, 
    * tRdata, * tUnrdata; 

  if((i >= colsy) || (j >= colsx)) return;
  if((!diagFlag) && (i == j)) {
    dfStats[idx] = real(0);
    // dpValues[idx] = real(0);
    return;
  }

  tObs = obs + (i*colsx+j)*obsDim;

  tRCoeffs = rCoeffs + i*rCoeffsDim;
  tRdata = rdata + i*rdataDim;
        
  tUnrCoeffs = unrCoeffs + (i*colsx+j)*unrCoeffsDim;
  tUnrdata = unrdata + (i*colsx+j)*unrdataDim;

  rSsq = unrSsq = real(0);
  for(k = 0; k < rows; k++) {
    unrEst = rEst = real(0);
    kobs = tObs[k];
    for(m = 0; m < rCols; m++)
      rEst += tRCoeffs[m] * tRdata[k+m*rows];
    for(m = 0; m < unrCols; m++) 
      unrEst += tUnrCoeffs[m] * tUnrdata[k+m*rows];
    rSsq   += (kobs - rEst) * (kobs - rEst);
    unrSsq += (kobs - unrEst) * (kobs - unrEst);

  }
  score = ((rSsq - unrSsq)*(frows-real(2)*fp-real(1))) / (fp*unrSsq);

  if(!isfinite(score))
    score = real(0);

  dfStats[idx] = score;
}
