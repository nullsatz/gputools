#ifndef _MI_H_
#define _MI_H_

template <typename T>
void bSplineMutualInfo(int nbins, int order, int nsamples,
                       int nx, const T * x,
                       int ny, const T * y,
                       T * out_mi);

#endif /* _MI_H_ */
