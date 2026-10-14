#ifndef _GRANGER_H_
#define _GRANGER_H_

template <typename T>
void granger(int rows, int cols,
             const T * y, int p, 
             T * fStats, T * pValues);
template <typename T>
void grangerxy(int rows,
               int colsx, const T * x,
               int colsy, const T * y,
               int p,
               T * fStats,
               T * pValues);

#endif /* _GRANGER_H_ */
