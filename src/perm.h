#ifndef _PERM_H_
#define _PERM_H_

// Permutation test of every correlation between the standardised columns
// of zx (n x p) and zy (n x q), column-major on the host (zy NULL: zx with
// itself). r0 gets the correlations (p x q). For each of the B
// permutations in perms (n x B, 0-based row indices), zy's rows are
// permuted and counts (p x q) incremented where the permuted |r| >=
// |r0| - eps; maxima[b] gets that permutation's largest |r| over the
// tests (the diagonal of zx with itself is no test).
template <typename T>
void permuteCorrelations(int n, int p, const T * zx, int q, const T * zy,
                         int B, const int * perms, double eps, double * r0,
                         int * counts, double * maxima);

#endif /* _PERM_H_ */
