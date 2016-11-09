extern "C" {
  // whichObs = 0 means everything
  // whichObs = 1 means pairwiseComplete
  void rpmcc(const int * whichObs,
             const float * samplesA, const int * numSamplesA,
             const float * samplesB, const int * numSamplesB, 
             const int * sampleSize, float * numPairs,
             float * correlations, float * signifs,
             const char ** kernel_src);

  void rformatInput(const int * images, 
                    const int * xcoords, const int * ycoords, const int * zcoords,
                    const int * mins, const int * maxes,
                    const float * evs, const int * numrows, const int * numimages, 
                    float * output);

  void rformatOutput(const int * imageList1, const int * numImages1, 
                     const int * imageList2, const int * numImages2, 
                     const int * structureid,
                     const double * cutCorrelation, const int * cutPairs,
                     const double * correlations, const double * signifs, 
                     const int * numPairs, double * results, int * nrows);

  void rsetDevice(const int * device);
  void rgetDevice(int * device);

  void rtestT(const float * pairs, const float * coeffs, const int * n, 
              float * ts, const char ** kernelSrc);
  void rhostT(const float * pairs, const float * coeffs, const int * n, 
              float * ts); 
  void rSignifFilter(const double * data, int * rows, double * results);
  void gSignifFilter(const float * data, int * rows, float * results,
                     const char ** kernelSrc);

  void RcublasPMCC(const float * samplesA, const int * numSamplesA,
                   const float * samplesB, const int * numSamplesB, 
                   const int * sampleSize, float * correlations,
                   const char ** kernelSrc);

  void RhostKendall(const float * X, const float * Y, const int * n, 
                    double * answer);
  void RpermHostKendall(const float * X, const int * nx, const float * Y, 
                        const int * ny, const int * sampleSize, double * answers);
  void RgpuKendall(const float * X, const int * nx, const float * Y, 
                   const int * ny, const int * sampleSize, double * answers,
                   const char ** kernel_src);

  void rgpuGranger(const int * rows, const int * colsy, const float * y, 
                   const int * p, float * fStats, float * pValues,
                   const char ** kernelSrc);
  void rgpuGrangerXY(const int * rows, const int * colsx, const float * x, 
                     const int * colsy, const float * y, const int * p, 
                     float * fStats, float * pValues,
                     const char ** kernelSrc);

  void Rdistclust(const char ** distmethod, const char ** clustmethod, 
                  const float * points, const int * numPoints, const int * dim,
                  int * merge, int * order, float * val,
                  const char ** distKernels,
                  const char ** clustKernels);
  void Rdistances(const float * points, const int * numPoints, 
                  const int * dim, float * distances, const char ** method,
                  const float * p, const char ** kernelSrc);
  void Rhcluster(const float * distMat, const int * numPoints, 
                 int * merge, int * order, float * val, const char ** method,
                 const char ** kernelSrc);

  void RgetQRDecomp(int * rows, int * cols, float * a, float * q, int * pivot,
                    int * rank, const char ** kernelSrc);
  void RqrSolver(int * rows, int * cols, float * matX, float * vectY, 
                 float * vectB, const char ** kernelSrc);

  void rGetQRDecompRR(const int * rows, const int * cols,
                      const double * tol, float * x, int * pivot,
                      double * qraux, int * rank,
                      const char ** kernelSrc);

  void rGetInverseFromQR(const int * rows, const int * cols, const float * q,
                         const float * r, float * inverse);
  void rSolveFromQR(const int * rows, const int * cols, const float * q,
                    const float * r, const float * y, float * b);

  void rBSplineMutualInfo(int * nBins, int * splineOrder, int * nsamples,
                          int * rowsA, const float * A,
                          int * rowsB, const float * B, 
                          float * mutualInfo,
                          const char ** kernelSrc);

  void RgpuLSFit(float *X, int *n, int *p, float *Y, int *nY,
                 double *tol, float *coeffs, float *resids, float *effects,
                 int *rank, int *pivot, double * qrAux, int useSingle,
                 const char ** kernelSrc);

  void setDevice(int * device);
}

void formatClustering(const int len, const int * sub,  const int * sup, 
                      int * merge, int * order);
void getPrintOrder(const int len, const int * merge, int * order);
void depthFirst(const int len, const int * merge, int level, int * otop, 
                int * order);
