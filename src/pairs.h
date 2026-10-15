#ifndef _PAIRS_H_
#define _PAIRS_H_

#include <vector>

// Pairs picked out of a correlation matrix: x column i, y column j
// (0-based), correlation r.
struct PairList {
  std::vector<int> i, j;
  std::vector<double> r;
};

// The correlation matrix zx' zy of unit-length columns (n x p and n x q,
// column-major, on the GPU; dZy == dZx for x with itself), computed tile by
// tile, tiles of tile x tile, so that it is never held whole.

// Every pair with |r| >= threshold (r >= threshold unless absolute); for x
// with itself, each pair once (i < j). Returns false, with out
// incomplete, if more than limit pairs pass.
template <typename T>
bool thresholdPairs(int n, int p, const T * dZx, int q, const T * dZy,
                    bool absolute, double threshold, int tile, double limit,
                    PairList & out);

// For each column j of y, the k columns of x most strongly correlated with
// it (by |r|, or r unless absolute), strongest first; for x with itself, j
// is not its own partner.
template <typename T>
void topPairs(int n, int p, const T * dZx, int q, const T * dZy,
              bool absolute, int k, int tile, PairList & out);

// thresholdPairs (mode 0) or topPairs (mode 1), then frees dZx and dZy;
// raises an R error if more than limit pairs pass the threshold.
template <typename T>
void selectPairs(int n, int p, T * dZx, int q, T * dZy, int mode,
                 double threshold, int k, bool absolute, int tile,
                 double limit, PairList & out);

// Replaces each column of the n x p matrix dZ (on the GPU, columns of unit
// length) by its residual from the m orthonormal columns of dQ, scaled back
// to unit length. lengths gets each residual's length before that scaling,
// relative to the column's, NaN where nothing but rounding error is left
// (the column is constant, or explained by the covariates); such a
// column becomes NaN and correlates with nothing.
template <typename T>
void residualise(int n, int p, T * dZ, int m, const T * dQ,
                 std::vector<double> & lengths);

#endif /* _PAIRS_H_ */
