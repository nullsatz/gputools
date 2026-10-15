#ifndef _SOLVE_H_
#define _SOLVE_H_

template <typename T>
void solveLinear(int n, int p, const T * a, int k, const T * b, T * x);

template <typename T>
void singularValues(int m, int n, const T * a, T * s, bool wantU, T * u,
                    bool wantV, T * v);

#endif /* _SOLVE_H_ */
