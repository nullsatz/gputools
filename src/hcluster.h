#ifndef HCLUSTER_H
#define HCLUSTER_H

/* Methods for hierarchical clustering */
typedef enum {
	SINGLE, COMPLETE, WPGMA, AVERAGE, MEDIAN, CENTROID, 
	FLEXIBLE_GROUP, FLEXIBLE, WARD, MCQUITTY
} hc_method;

void hcluster(const float * dist, size_t dist_pitch, size_t n,
	int * sub, int * sup, float * val, hc_method method,
	const float lambda = 0.5, const float beta = 0.5);

void hclusterPreparedDistances(float * gpuDist, size_t pitch_dist_d, size_t n,
	      int * sub, int * sup, float * val, hc_method method,
	      const float lambda, const float beta);

#endif // HCLUSTER_H
