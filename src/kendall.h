#ifndef _KENDALL_H_
#define _KENDALL_H_

#include <cstddef>

template <typename T>
void rankCorrelation(const char * kernel, const T * x, size_t nx,
                     const T * y, size_t ny, size_t sampleSize,
                     double * results, double * counts);

#endif /* _KENDALL_H_ */
