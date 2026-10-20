// B-spline mutual information of every pair of variables: Daub et al.
// (2004), with the weights computed in double from the double data and
// the joint distributions of every pair from one matrix product per tile
// (see inst/cuda/mi.cu for the kernels and the formulas). T is the
// precision of the products; the method and its guards follow the Python
// package widestats (mi.py).

#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda_runtime.h"

#include "R.h"

#include "cudaUtils.h"
#include "gpublas.h"
#include "mi.h"

namespace {

// the weights of nvars variables, n x nvars*bins in T on the device
template <typename T>
T * weights(int n, int nvars, int bins, int order, const double * x)
{
  double * dX = gpuAlloc<double>((size_t) n * nvars);
  CUDART_CALL(cudaMemcpy(dX, x, (size_t) n * nvars * sizeof(double),
                         cudaMemcpyHostToDevice));
  size_t count = (size_t) n * nvars * bins;
  double * dW = gpuAlloc<double>(count);
  int threads = 128;
  {
    int nn = n, nv = nvars, nb = bins, no = order;
    void * args[] = { &nn, &nv, &nb, &no, &dX, &dW };
    cudaLaunch("miWeights", args, dim3((nvars + threads - 1) / threads),
               dim3(threads), Precision::Double);
  }
  cudaFree(dX);
  if (sizeof(T) == sizeof(double)) return (T *) dW;
  // the product in single: narrow the weights once
  T * dWs = gpuAlloc<T>(count);
  {
    size_t cnt = count;
    void * args[] = { &dW, &dWs, &cnt };
    cudaLaunch("miNarrow", args,
               dim3((unsigned) ((count + 255) / 256)), dim3(256),
               Precision::Double);
  }
  cudaFree(dW);
  return dWs;
}

} // namespace

template <typename T>
void bSplineMutualInfo(int bins, int order, int n, int nx, const double * x,
                       int ny, const double * y, double * out)
{
  const bool self = y == NULL || y == x;
  if (self) { y = x; ny = nx; }
  T * dWx = weights<T>(n, nx, bins, order, x);
  T * dWy = self ? dWx : weights<T>(n, ny, bins, order, y);
  T * dMi = gpuAlloc<T>((size_t) nx * ny);
  // a tile's product holds (tile*bins)^2 entries: about 2^24 of them
  int tile = std::max(1, (int) (4096 / bins));
  T * dJ = gpuAlloc<T>((size_t) tile * bins * tile * bins);
  T invn = (T) (1.0 / n);
  for (int i0 = 0; i0 < nx; i0 += tile) {
    int tx = std::min(tile, nx - i0);
    for (int j0 = 0; j0 < ny; j0 += tile) {
      int ty = std::min(tile, ny - j0);
      int ldj = tx * bins;
      // every pair's joint distribution (times n) in this tile: a bins x
      // bins block of W_x' W_y per pair
      blas::gemm('T', 'N', tx * bins, ty * bins, n, T(1),
                 dWx + (size_t) i0 * bins * n, n,
                 dWy + (size_t) j0 * bins * n, n, T(0), dJ, ldj);
      int nb = bins, ttx = tx, tty = ty, ii0 = i0, jj0 = j0, ldm = ny;
      size_t pairs = (size_t) tx * ty;
      void * args[] = { &nb, &ttx, &tty, &dJ, &ldj, &invn, &ii0, &jj0, &dMi,
                        &ldm };
      cudaLaunch("miPairs", args, dim3((unsigned) ((pairs + 255) / 256)),
                 dim3(256), precisionOf<T>());
    }
  }
  std::vector<T> h((size_t) nx * ny);
  CUDART_CALL(cudaMemcpy(h.data(), dMi, h.size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
  cudaFree(dJ); cudaFree(dMi);
  if (!self) cudaFree(dWy);
  cudaFree(dWx);
  // out[j + i*ny] is the information of y_j with x_i. A constant variable
  // shares no information: exactly 0, not rounding; an estimate of 0 can
  // round just below it
  std::vector<char> cx(nx, 0), cy(ny, 0);
  for (int i = 0; i < nx; i++) {
    const double * c = x + (size_t) i * n;
    cx[i] = std::equal(c + 1, c + n, c);
  }
  for (int j = 0; j < ny; j++) {
    const double * c = y + (size_t) j * n;
    cy[j] = self ? cx[j] : std::equal(c + 1, c + n, c);
  }
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      size_t e = (size_t) j + (size_t) i * ny;
      out[e] = (cx[i] || cy[j]) ? 0.0 : std::max(0.0, (double) h[e]);
    }
}

template void bSplineMutualInfo<float>(int, int, int, int, const double *,
                                       int, const double *, double *);
template void bSplineMutualInfo<double>(int, int, int, int, const double *,
                                        int, const double *, double *);
