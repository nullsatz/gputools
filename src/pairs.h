#ifndef _PAIRS_H_
#define _PAIRS_H_

#include <vector>

// Pairs picked out of a correlation matrix: x column i, y column j
// (0-based), correlation r.
struct PairList {
  std::vector<int> i, j;
  std::vector<double> r;
};

// The correlation matrix zx' zy of standardised columns (n x p and n x q,
// column-major on the host; zy NULL for zx with itself), computed tile by
// tile on the GPU, tiles of tile x tile, so that it is never held whole.

// Every pair with |r| >= threshold (r >= threshold unless absolute); for x
// with itself, each pair once (i < j). Raises an R error if more than
// limit pairs pass.
template <typename T>
void thresholdPairs(int n, int p, const T * zx, int q, const T * zy,
                    bool absolute, double threshold, int tile, double limit,
                    PairList & out);

// For each column j of y, the k columns of x most strongly correlated with
// it (by |r|, or r unless absolute), strongest first; for x with itself, j
// is not its own partner.
template <typename T>
void topPairs(int n, int p, const T * zx, int q, const T * zy,
              bool absolute, int k, int tile, PairList & out);

#endif /* _PAIRS_H_ */
