#ifndef DISTANCE_H
#define DISTANCE_H

// Methods for computing the distance
typedef enum {
	EUCLIDEAN,	MAXIMUM,	MANHATTAN, CANBERRA, 
	BINARY, 	MINKOWSKI //	,DOT
} dist_method;

/* Calculate the distance matrix for vectors in group a and vectors in group b.
 *
 * The format for vg_a and vg_b is as follows:
 *   There are n_* vectors, each of dimensionality k.  They are stored in
 *   row major order with a row (or pitch) being pitch_* bytes.
 * The calculated distances are stored in d such that the distance between
 * vectors indexed a and b is located in d[b * pitch_d / sizeof(float) + a].
 * The user is responsible for
 * allocating storage for d.  It should be at least n_a * n_b * sizeof(float)
 * bytes.  The pitch_d argument is the same as for vg_*.
 * The method used to calculate the distance is dist_method.
 * 
 * The argument p is optional and used for the Minkowski method.
 *
 * This function may run on the CPU or GPU depending on the size and
 * number of vectors.  Good data alignment will increase performance.
 */
template <typename T>
void distance(const T * vg_a, size_t pitch_a, size_t n_a,
	      const T * vg_b, size_t pitch_b, size_t n_b,
	      size_t k,
	      T * d, size_t pitch_d,
	      dist_method method,
	      T p);

/* This function should be used to calculate a distance matrix when 
 * the data is already stored on the GPU.  vg_a, vg_b, and
 * d should already be allocated and initialized on the device.
 * This function does not allocate or free any storage on the device.
 */
template <typename T>
void distance_device(const T * vg_a, size_t pitch_a, size_t n_a,
		     const T * vg_b, size_t pitch_b, size_t n_b,
		     size_t k,
		     T * d, size_t pitch_d,
		     dist_method method,
		     T p);

template <typename T>
void distanceLeaveOnGpu(dist_method method, T p, const T * points, 
	size_t dim, size_t numPoints, T ** gpuDistances, 
                        size_t * pitchDistances);


#endif // DISTANCE_H
