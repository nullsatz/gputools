#ifndef _KENDALL_H_
#define _KENDALL_H_

template <typename T>
void masterKendall(const T * x,  size_t nx, 
                   const T * y, size_t ny,
                   size_t sampleSize,
                   double * results);

#endif /* _KENDALL_H_ */
