#ifndef _MI_H_
#define _MI_H_

void bSplineMutualInfo(int nbins, int order, int nsamples,
                       int nx, const float * x,
                       int ny, const float * y,
                       float * out_mi);

#endif /* _MI_H_ */
