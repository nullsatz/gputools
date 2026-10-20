// B-spline mutual information (Daub et al. 2004) of every pair of
// variables, as src/mi.cpp drives it:
//
//   - each variable's values get B-spline weights over `bins` bins
//     (miWeights, always in double from the double data: rounding the data
//     to single first moves a value on a knot, as discrete data put them,
//     into the neighbouring bin), laid out as W (n x nvars*bins,
//     column-major: sample k of variable v's bin i at W[k + (v*bins + i)*n])
//     so that one matrix product W_x' W_y / n (cuBLAS, from the host) holds
//     the joint distribution of every pair as a bins x bins block;
//   - miPairs reads each block and computes the mutual information in log
//     form, sum P log2(P / (P_a P_b)) with the marginals P_a, P_b the
//     block's own row and column sums: the same as H_a + H_b - H_ab, but
//     without subtracting large entropies, and consistent with any rounding
//     in the product. 0 log 0 counts as 0.
//
// bins + order - 1 is at most MI_MAXSLOTS (the R code limits bins to 64).

#define MI_MAXSLOTS 128

// the knot vector 0 (order times), 1, ..., bins - order, span (order
// times), span = bins - order + 1; knot i of bins + order
__device__ double miKnot(int i, int bins, int order)
{
  if (i < order) return 0.0;
  if (i <= bins - 1) return (double) (i - order + 1);
  return (double) (bins - order + 1);
}

// Each variable's weights: one thread per variable, over its n samples.
// A constant variable is put wholly at the domain's start (every value in
// the first bin).
__global__ void miWeights(int n, int nvars, int bins, int order,
                          const double * x, double * w)
{
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= nvars) return;
  const double * col = x + (size_t) v * n;
  double lo = col[0], hi = col[0];
  for (int k = 1; k < n; k++) {
    if (col[k] < lo) lo = col[k];
    if (col[k] > hi) hi = col[k];
  }
  double width = hi > lo ? hi - lo : 1.0;
  int span = bins - order + 1, slots = bins + order - 1;
  double wk[MI_MAXSLOTS];
  for (int k = 0; k < n; k++) {
    double z = (col[k] - lo) / width * span;
    // order 1: the knot interval holding the value, among the non-empty
    // ones order - 1 .. bins - 1; the far end of the domain belongs to the
    // last
    int cell = (int) floor(z);
    if (cell < 0) cell = 0;
    if (cell > span - 1) cell = span - 1;
    cell += order - 1;
    for (int i = 0; i < slots; i++) wk[i] = 0.0;
    wk[cell] = 1.0;
    // Cox-de Boor: B_{i,k} = (z - t_i)/(t_{i+k-1} - t_i) B_{i,k-1}
    //   + (t_{i+k} - z)/(t_{i+k} - t_{i+1}) B_{i+1,k-1}, 0/0 as 0
    for (int kk = 2; kk <= order; kk++) {
      int count = bins + order - kk;
      for (int i = 0; i < count; i++) {
        double ti = miKnot(i, bins, order), tik = miKnot(i + kk, bins, order);
        double lden = miKnot(i + kk - 1, bins, order) - ti;
        double rden = tik - miKnot(i + 1, bins, order);
        double left = lden > 0 ? (z - ti) / lden * wk[i] : 0.0;
        double right = rden > 0 ? (tik - z) / rden * wk[i + 1] : 0.0;
        wk[i] = left + right;
      }
    }
    for (int i = 0; i < bins; i++)
      w[k + ((size_t) v * bins + i) * n] = wk[i];
  }
}

// Single-precision copy of the weights, for the product in single.
__global__ void miNarrow(const double * in, float * out, size_t count)
{
  size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (e < count) out[e] = (float) in[e];
}

// The mutual information of each pair of a tile from the product j
// (tx*bins x ty*bins, leading dimension ldj) of their weights, scaled by
// invn = 1/n to the joint distribution. Writes mi[(j0 + b) + (i0 + a)*ldm]
// for x variable a and y variable b of the tile, in bits.
__global__ void miPairs(int bins, int tx, int ty, const real * j, int ldj,
                        real invn, int i0, int j0, real * mi, int ldm)
{
  size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
  if (e >= (size_t) tx * ty) return;
  int a = e % tx, b = e / tx;
  const real * block = j + (size_t) a * bins + (size_t) b * bins * ldj;
  real rows[MI_MAXSLOTS], cols[MI_MAXSLOTS];
  for (int i = 0; i < bins; i++) { rows[i] = real(0); cols[i] = real(0); }
  for (int c = 0; c < bins; c++)
    for (int r = 0; r < bins; r++) {
      real p = block[r + (size_t) c * ldj] * invn;
      rows[r] += p;
      cols[c] += p;
    }
  // in logs, each clamped below at the smallest normal number: where an
  // entry is positive its row and column sums are at least as large, so
  // every log is finite
  const real tiny = sizeof(real) == 4 ? real(1.17549435e-38)
                                      : real(2.2250738585072014e-308);
  real sum = real(0);
  for (int c = 0; c < bins; c++)
    for (int r = 0; r < bins; r++) {
      real p = block[r + (size_t) c * ldj] * invn;
      if (p > real(0)) {
        real lr = rows[r] > tiny ? rows[r] : tiny;
        real lc = cols[c] > tiny ? cols[c] : tiny;
        sum += p * (log2(p) - log2(lr) - log2(lc));
      }
    }
  mi[(size_t) (j0 + b) + (size_t) (i0 + a) * ldm] = sum;
}
