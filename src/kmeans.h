#ifndef _KMEANS_H_
#define _KMEANS_H_

// Lloyd's k-means of the n points in x (n x d, column-major on the host)
// from the k starting centres in centers (k x d, column-major), which it
// updates in place. cluster gets each point's cluster (1-based), sizes
// each cluster's number of points, iterations the number of assignment
// passes (maxiter + 1 if it did not converge), as R's kmeans_Lloyd.
template <typename T>
void lloyd(int n, int d, const T * x, int k, double * centers, int maxiter,
           int * cluster, int * sizes, int * iterations);

#endif /* _KMEANS_H_ */
