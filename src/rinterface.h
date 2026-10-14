#ifndef _RINTERFACE_H_
#define _RINTERFACE_H_

#include "Rinternals.h"

#include "distance.h"
#include "hcluster.h"

dist_method getDistEnum(const char * methodStr);
hc_method getClusterEnum(const char * methodStr);

void formatClustering(const int len, const int * sub, const int * sup,
                      int * merge, int * order);
void getPrintOrder(const int len, const int * merge, int * order);
void depthFirst(const int len, const int * merge, int level, int * otop,
                int * order);

extern "C" {
  SEXP gputoolsSetKernelDir(SEXP dir);
  SEXP gputoolsUnloadKernels();
}

#endif /* _RINTERFACE_H_ */
