#ifndef HCLUSTER_H
#define HCLUSTER_H

/* Methods for hierarchical clustering */
typedef enum {
  SINGLE,
  COMPLETE,
  WPGMA,
  AVERAGE,
  MEDIAN,
  CENTROID, 
  FLEXIBLE_GROUP,
  FLEXIBLE,
  WARD,
  MCQUITTY
} hc_method;

template <typename T>
void hcluster(const T * dist, size_t dist_pitch, size_t n,
              int * sub, int * sup,
              T * val,
              hc_method method,
              T lambda = 0.5, T beta = 0.5);

template <typename T>
void hclusterPreparedDistances(T * gpuDist, size_t pitch_dist_d, size_t n,
                               int * sub, int * sup,
                               T * val,
                               hc_method method,
                               T lambda, T beta);

#endif // HCLUSTER_H
