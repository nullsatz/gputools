#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "cuda_runtime.h"

#include "R.h"

#include "cudaUtils.h"
#include "gpublas.h"

#include "pairs.h"

namespace {

// zx and zy, already on the GPU (dZy == dZx for x with itself), and the
// tile of their product being worked on
template <typename T>
struct Tiles {
  int n, p, q, tile;
  bool same;
  const T * dZx, * dZy;
  T * dR;

  Tiles(int n, int p, const T * dZx, int q, const T * dZy, int tile)
    : n(n), p(p), q(q), tile(tile), same(dZy == dZx), dZx(dZx), dZy(dZy),
      dR(gpuAlloc<T>((size_t) tile * tile)) {}

  void release()
  {
    cudaFree(dR);
    dR = NULL;
  }

  // the correlations of x columns i0 .. i0 + ta - 1 with y columns
  // j0 .. j0 + tb - 1, into dR (ta x tb)
  void product(int i0, int ta, int j0, int tb)
  {
    blas::gemm('T', 'N', ta, tb, n, T(1), dZx + (size_t) i0 * n, n,
               dZy + (size_t) j0 * n, n, T(0), dR, ta);
  }
};

} // namespace

template <typename T>
bool thresholdPairs(int n, int p, const T * dZx, int q, const T * dZy,
                    bool absolute, double threshold, int tile, double limit,
                    PairList & out)
{
  Tiles<T> t(n, p, dZx, q, dZy, tile);
  size_t cap = (size_t) tile * tile;
  int * dCount = gpuAlloc<int>(1), * dI = gpuAlloc<int>(cap),
    * dJ = gpuAlloc<int>(cap);
  T * dV = gpuAlloc<T>(cap);
  std::vector<int> hI(cap), hJ(cap);
  std::vector<T> hV(cap);
  int lower = t.same;
  int absol = absolute;
  T thr = (T) threshold;
  bool tooMany = false;

  for (int j0 = 0; j0 < q && !tooMany; j0 += tile) {
    int tb = std::min(tile, q - j0);
    for (int i0 = 0; i0 < p && !tooMany; i0 += tile) {
      int ta = std::min(tile, p - i0);
      if (lower && i0 >= j0 + tb - 1) break;   // no i < j left in this column
      t.product(i0, ta, j0, tb);
      CUDART_CALL(cudaMemset(dCount, 0, sizeof(int)));
      size_t entries = (size_t) ta * tb;
      int threads = 256;
      dim3 grid((unsigned) ((entries + threads - 1) / threads)),
        block(threads);
      void * args[] = { &t.dR, &ta, &tb, &i0, &j0, &lower, &absol, &thr,
                        &dCount, &dI, &dJ, &dV };
      cudaLaunch("gpuThresholdPairs", args, grid, block, precisionOf<T>());
      int count = 0;
      CUDART_CALL(cudaMemcpy(&count, dCount, sizeof(int),
                             cudaMemcpyDeviceToHost));
      if (out.r.size() + (double) count > limit) {
        tooMany = true;
        break;
      }
      CUDART_CALL(cudaMemcpy(hI.data(), dI, count * sizeof(int),
                             cudaMemcpyDeviceToHost));
      CUDART_CALL(cudaMemcpy(hJ.data(), dJ, count * sizeof(int),
                             cudaMemcpyDeviceToHost));
      CUDART_CALL(cudaMemcpy(hV.data(), dV, count * sizeof(T),
                             cudaMemcpyDeviceToHost));
      out.i.insert(out.i.end(), hI.begin(), hI.begin() + count);
      out.j.insert(out.j.end(), hJ.begin(), hJ.begin() + count);
      out.r.insert(out.r.end(), hV.begin(), hV.begin() + count);
    }
  }

  cudaFree(dCount); cudaFree(dI); cudaFree(dJ); cudaFree(dV);
  t.release();
  return !tooMany;
}

template <typename T>
void topPairs(int n, int p, const T * dZx, int q, const T * dZy,
              bool absolute, int k, int tile, PairList & out)
{
  Tiles<T> t(n, p, dZx, q, dZy, tile);
  int n2 = 1;
  while (n2 < std::min(tile, p)) n2 <<= 1;
  int kt = std::min(k, std::min(tile, p));   // per tile: at most ta useful
  T * dKeys = gpuAlloc<T>((size_t) tile * n2), * dOutR =
    gpuAlloc<T>((size_t) tile * kt);
  int * dPos = gpuAlloc<int>((size_t) tile * n2), * dOutI =
    gpuAlloc<int>((size_t) tile * kt);
  std::vector<int> hI((size_t) tile * kt);
  std::vector<T> hR((size_t) tile * kt);
  int lower = t.same, absol = absolute;

  struct Cand { double key; int i; double r; };
  auto stronger = [](const Cand & a, const Cand & b) {
    return a.key > b.key || (a.key == b.key && a.i < b.i);
  };

  for (int j0 = 0; j0 < q; j0 += tile) {
    int tb = std::min(tile, q - j0);
    std::vector<std::vector<Cand>> best(tb);
    for (int i0 = 0; i0 < p; i0 += tile) {
      int ta = std::min(tile, p - i0);
      t.product(i0, ta, j0, tb);
      int ka = std::min(kt, ta);
      dim3 grid(tb), block(n2 < 512 ? n2 : 512);
      void * args[] = { &t.dR, &ta, &n2, &i0, &j0, &lower, &absol, &ka,
                        &dKeys, &dPos, &dOutI, &dOutR };
      cudaLaunch("gpuTopPerColumn", args, grid, block, precisionOf<T>());
      CUDART_CALL(cudaMemcpy(hI.data(), dOutI, (size_t) tb * ka * sizeof(int),
                             cudaMemcpyDeviceToHost));
      CUDART_CALL(cudaMemcpy(hR.data(), dOutR, (size_t) tb * ka * sizeof(T),
                             cudaMemcpyDeviceToHost));
      // merge this tile's best into each column's running best k
      for (int b = 0; b < tb; ++b) {
        std::vector<Cand> & c = best[b];
        for (int s = 0; s < ka; ++s) {
          int i = hI[(size_t) b * ka + s];
          if (i < 0) break;   // the rest of this column is unusable too
          double r = hR[(size_t) b * ka + s];
          c.push_back({ absolute ? std::fabs(r) : r, i, r });
        }
        if ((int) c.size() > k) {
          std::partial_sort(c.begin(), c.begin() + k, c.end(), stronger);
          c.resize(k);
        }
      }
    }
    for (int b = 0; b < tb; ++b) {
      std::vector<Cand> & c = best[b];
      std::sort(c.begin(), c.end(), stronger);
      for (const Cand & x : c) {
        out.i.push_back(x.i);
        out.j.push_back(j0 + b);
        out.r.push_back(x.r);
      }
    }
  }

  cudaFree(dKeys); cudaFree(dOutR); cudaFree(dPos); cudaFree(dOutI);
  t.release();
}

template <typename T>
void selectPairs(int n, int p, T * dZx, int q, T * dZy, int mode,
                 double threshold, int k, bool absolute, int tile,
                 double limit, PairList & out)
{
  bool ok = true;
  if (mode == 0)
    ok = thresholdPairs<T>(n, p, dZx, q, dZy, absolute, threshold, tile,
                           limit, out);
  else
    topPairs<T>(n, p, dZx, q, dZy, absolute, k, tile, out);
  cudaFree(dZx);
  if (dZy != dZx) cudaFree(dZy);
  if (!ok)
    error("gputools: more than %.0f pairs pass the threshold; raise the "
          "threshold, or raise limit if you want them all", limit);
}

template <typename T>
void residualise(int n, int p, T * dZ, int m, const T * dQ,
                 std::vector<double> & lengths)
{
  if (m > 0) {
    // Z -= Q (Q' Z)
    T * dW = gpuAlloc<T>((size_t) m * p);
    blas::gemm('T', 'N', m, p, n, T(1), dQ, n, dZ, n, T(0), dW, m);
    blas::gemm('N', 'N', n, p, m, T(-1), dQ, n, dW, m, T(1), dZ, n);
    cudaFree(dW);
  }

  double * dNorms = gpuAlloc<double>(p);
  dim3 grid(p), block(256);
  void * normArgs[] = { &dZ, &n, &dNorms };
  cudaLaunch("gpuColumnNorms", normArgs, grid, block, precisionOf<T>());
  lengths.resize(p);
  CUDART_CALL(cudaMemcpy(lengths.data(), dNorms, p * sizeof(double),
                         cudaMemcpyDeviceToHost));
  cudaFree(dNorms);

  // what the covariates leave of a column is rounding error if it is this
  // small a fraction of the unit-length column
  double tol = sizeof(T) == sizeof(float) ? 1e-5 : 1e-10;
  std::vector<T> scale(p);
  for (int j = 0; j < p; ++j) {
    bool usable = lengths[j] > tol;   // false for NaN too
    scale[j] = usable ? (T) (1 / lengths[j]) : std::numeric_limits<T>::quiet_NaN();
    if (!usable) lengths[j] = NAN;
  }
  T * dScale = gpuAlloc<T>(p);
  CUDART_CALL(cudaMemcpy(dScale, scale.data(), p * sizeof(T),
                         cudaMemcpyHostToDevice));
  size_t entries = (size_t) n * p;
  dim3 sgrid((unsigned) ((entries + 255) / 256)), sblock(256);
  void * scaleArgs[] = { &dZ, &n, &p, &dScale };
  cudaLaunch("gpuScaleColumns", scaleArgs, sgrid, sblock, precisionOf<T>());
  cudaFree(dScale);
}

template void residualise<float>(int, int, float *, int, const float *,
  std::vector<double> &);
template void residualise<double>(int, int, double *, int, const double *,
  std::vector<double> &);
template void selectPairs<float>(int, int, float *, int, float *, int,
  double, int, bool, int, double, PairList &);
template void selectPairs<double>(int, int, double *, int, double *, int,
  double, int, bool, int, double, PairList &);
template bool thresholdPairs<float>(int, int, const float *, int,
  const float *, bool, double, int, double, PairList &);
template bool thresholdPairs<double>(int, int, const double *, int,
  const double *, bool, double, int, double, PairList &);
template void topPairs<float>(int, int, const float *, int, const float *,
  bool, int, int, PairList &);
template void topPairs<double>(int, int, const double *, int,
  const double *, bool, int, int, PairList &);
