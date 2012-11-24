#include <R.h>
#include <cula.h>
#include <cublas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int min_JM (int *, int *);
int max_JM (int *, int *);

void rowcentre_JM (float *, int, int);
void colstandard_JM (float *, int, int);
void rowstd_JM (float *, int, int, int);
void transpose_mat_JM (float *, int *, int *, float *);
void mmult_JM (float *, int, int, float *, int, int, float *);
void orthog_mat_JM (float *, int, float *);
void gramsch_JM (float *, int, int, int);
void svd_JM (float *, int *, int *, float *, float *, float *);

void Symm_logcosh_JM (float *, int, float *, int, int, float, float *, float *);
void Symm_exp_JM (float *, int, float *, int, int, float, float *, float *);
void Def_logcosh_JM (float *, int, float *, int, int, float, float *);
void Def_exp_JM (float *, int, float *, int, int, float, float *);
void calc_A_JM(float*, float*, float*, int*, int*, int*, float*, float*);
void calc_K_JM(float*, int*, int*, float*);
/*
void F77_NAME (sgesdd) (char *, int *, int *, float *, int *, float *,
			float *, int *, float *, int *, float *, int *,
			int *, int *); */

void
rowcentre_JM (float *ans, int n, int p)
{
/*  mean centres nxp matrix ans */
	double tmp;
	int i, j;
	for (i = 0; i < n; i++) {
		tmp = 0;
		for (j = 0; j < p; j++) {
			tmp = tmp + ((double) ans[p * i + j]) / p;
		}
		for (j = 0; j < p; j++) {
			ans[p * i + j] -= (float) tmp;
		}
	}
}

void
colstandard_JM (float *ans, int n, int p)
{
/*  transform columns of nxp matrix ans to have zero mean and unit variance */
	double tmp[2];
	double tmp1;
	int i, j;
	for (i = 0; i < p; i++) {
		tmp[0] = 0;
		tmp[1] = 0;

		for (j = 0; j < n; j++) {
			tmp[0] += (double) ans[p * j + i];
			tmp[1] += ((double) ans[p * j + i]) * ((double) ans[p * j + i]);
		}

		tmp[0] = tmp[0] / n;
		tmp1 = (tmp[1] - n * (tmp[0]) * (tmp[0])) / (n - 1);

		tmp[1] = sqrt (tmp1);
		for (j = 0; j < n; j++) {
			ans[p * j + i] =
				(float) ((((double) ans[p * j + i]) - tmp[0]) / tmp[1]);
		}
	}
}


void
svd_JM (float *mat, int *n, int *p, float *u, float *d, float *v)
{
	/*  calculates svd decomposition of nxp matrix mat */
	/*    mat is a pointer to an nxp array of floats */
	/*    n is a pointer to an integer specifying the no. of rows of mat */
	/*    p is a pointer to an integer specifying the no. of cols of mat */
	/*    u is a pointer to a float array of dimension (n,n) */
	/*    d is a pointer to a float array of dimension min(n,p) */
	/*    v is a pointer to a float array of dimension (p,p) */


	//int
	//	info, iwork_size, *iwork, lwork,
	//	a, b;
	float
		// *work,
		*mat1, *u1, *v1;
	char
		jobz = 'A';

	// iwork_size = 8 * min_JM (n, p);

	// a = max_JM(n,p);
	// b = 4 * min_JM(n,p) * min_JM(n,p) + 4 * min_JM(n,p);
	// lwork= 3 * min_JM(n,p) * min_JM(n,p) + max_JM(&a, &b);

	// work = Calloc (lwork, float);
	// iwork = Calloc (iwork_size, int);
	mat1 = Calloc ((*n) * (*p), float);
	u1 = Calloc ((*n) * (*n), float);
	v1 = Calloc ((*p) * (*p), float);

	transpose_mat_JM (mat, n, p, mat1);

//	F77_CALL (sgesdd) (&jobz, n, p, mat1, n, d, u1, n, v1, p, work,
//			   &lwork, iwork, &info);

	culaSgesvd(jobz, jobz, *n, *p, mat1, *n, d, u1, *n, v1, *p);

	transpose_mat_JM (u1, n, n, u);

	transpose_mat_JM (v1, p, p, v);

	Free (mat1);
	Free (u1);
	Free (v1);
	// Free (work);
	// Free (iwork);
}

/* void */
/* svd_old_JM (float *mat, int *n, int *p, float *u, float *d, float *v) */
/* { */

/*     /\*  calculates svd decomposition of nxp matrix mat *\/ */
/*     /\*    mat is a pointer to an nxp array of floats *\/ */
/*     /\*    n is a pointer to an integer specifying the no. of rows of mat *\/ */
/*     /\*    p is a pointer to an integer specifying the no. of cols of mat *\/ */
/*     /\*    u is a pointer to a float array of dimension (n,n) *\/ */
/*     /\*    d is a pointer to a float array of dimension min(n,p) *\/ */
/*     /\*    v is a pointer to a float array of dimension (p,p) *\/ */


/*     int info, lwork, i, j; */
/*     float *work, *mat1, *u1, *v1; */
/*     char jobu = 'A', jobvt = 'A'; */

/*     i = 3 * min_JM (n, p) + max_JM (n, p); */
/*     j = 5 * min_JM (n, p); */
/*     lwork = 10 * max_JM (&i, &j); */

/*     work = Calloc (lwork, float); */
/*     mat1 = Calloc ((*n) * (*p), float); */
/*     u1 = Calloc ((*n) * (*n), float); */
/*     v1 = Calloc ((*p) * (*p), float); */

/*     transpose_mat_JM (mat, n, p, mat1); */

/*     F77_CALL (sgesvd) (&jobu, &jobvt, n, p, mat1, n, d, u1, n, v1, p, work, */
/* 		       &lwork, &info); */

/*     transpose_mat_JM (u1, n, n, u); */

/*     transpose_mat_JM (v1, p, p, v); */


/*     Free (mat1); */
/*     Free (u1); */
/*     Free (v1); */
/*     Free (work); */

/* } */

void
transpose_mat_JM (float *mat, int *n, int *p, float *ans)
{
/*    transpose nxp matrix mat */
	int i, j;

	for (i = 0; i < *n; i++) {
		for (j = 0; j < *p; j++) {
			*(ans + j * (*n) + i) = *(mat + i * (*p) + j);
		}
	}
}


int
min_JM (int *a, int *b)
{
/*  find minimum of a and b */
	int ans;

	ans = *b;
	if (*a < *b)
		ans = *a;

	return ans;
}

int
max_JM (int *a, int *b)
{
/*  find maximum of a and b */

	int ans;

	ans = *b;
	if (*a > *b)
		ans = *a;

	return ans;
}

/*  sgemm -- Level 3 Blas routine. */
/*  -- Written on 8-February-1989. */
/*	 Jack Dongarra, Argonne National Laboratory. */
/*	 Iain Duff, AERE Harwell. */
/*	 Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*	 Sven Hammarling, Numerical Algorithms Group Ltd. */

void sgemm(char transa, char transb, int m, int n, int k,
	float alpha, const float * a, int lda, const float * b, int ldb,
	float beta, float * c, int ldc)
{
	int
		i, j, l,
		nota, notb,
		ncola,
		nrowa, nrowb;

	float temp;

/*	 Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not */
/*	 transposed and set  NROWA, NCOLA and  NROWB  as the number of rows */
/*	 and  columns of  A  and the  number of  rows  of  B  respectively. */

	/* Function Body */
	nota = (transa == 'N');
	notb = (transb == 'N');
	if (nota) {
		nrowa = m;
		ncola = k;
	} else {
		nrowa = k;
		ncola = m;
	}

	if (notb)
		nrowb = k;
	else
		nrowb = n;

/*	 Test the input parameters. */

	if (!nota && !(transa == 'C') && !(transa == 'T'))
		error("argument 1 is bad");
	else if (!notb && !(transb == 'C') && !(transb == 'T'))
		error("argument 2 is bad");
	else if (m < 0)
		error("argument 3 is bad");
	else if (n < 0)
		error("argument 4 is bad");
	else if (k < 0)
		error("argument 5 is bad");
	else if(a == NULL)
		error("argument 7 is bad");
	else if (lda < max(1,nrowa))
		error("argument 8 is bad");
	else if(b == NULL)
		error("argument 9 is bad");
	else if (ldb < max(1,nrowb))
		error("argument 10 is bad");
	else if(c == NULL)
		error("argument 12 is bad");
	else if (ldc < max(1, m))
		error("argument 13 is bad");

	if (m == 0 || n == 0 || (alpha == 0.f || k == 0) && beta == 1.f)
		return;

	float * colj;
	if (alpha == 0.f) {
		if (beta == 0.f) {
			for (j = 0; j < n; j++)
				memset(c + j * ldc, 0, m * sizeof(float));
		} else {
			for (j = 0; j < n; j++) {
				for (i = 0; i < m; i++)
					c[i + j * ldc] = beta * c[i + j * ldc];
			}
		}
		return;
	}

	if (notb) {
		if (nota) { /* Form  C := alpha*A*B + beta*C. */
			for (j = 0; j < n; j++) {
				if (beta == 0.f)
					memset(c + j * ldc, 0, m * sizeof(float));
				else if (beta != 1.f) {
					for (i = 0; i < m; i++)
						c[i + j * ldc] = beta * c[i + j * ldc];
				}
				for (l = 0; l < k; l++) {
					if (b[l + j * ldb] != 0.f) {
						temp = alpha * b[l + j * ldb];
						for (i = 0; i <= m; i++)
							c[i + j * ldc] += temp * a[i + l * lda];
					}
				}
			}
		} else { /* Form  C := alpha*A'*B + beta*C */
			for (j = 0; j < n; j++) {
				for (i = 0; i < m; i++) {
					temp = 0.f;
					for (l = 0; l < k; l++)
						temp += a[l + i * lda] * b[l + j * ldb];

					if (beta == 0.f)
						c[i + j * ldc] = alpha * temp;
					else
						c[i + j * ldc] = alpha * temp + beta * c[ i + j * ldc];
				}
			}
		}
	} else {
		if (nota) { /* Form  C := alpha*A*B' + beta*C */
			for (j = 0; j < n; j++) {
				if (beta == 0.f)
					memset(c + j * ldc, 0, m * sizeof(float));
				else if (beta != 1.f) {
					for (i = 0; i < m; i++)
						c[i + j * ldc] = beta * c[i + j * ldc];
				}
				for (l = 0; l < k; l++) {
					if (b[j + l * ldb] != 0.f) {
						temp = alpha * b[j + l * ldb];
						for (i = 0; i < m; i++)
							c[i + j * ldc] += temp * a[i + l * lda];
					}
				}
			}
		} else { /* Form  C := alpha*A'*B' + beta*C */
			for (j = 0; j < n; j++) {
				for (i = 0; i < m; i++) {
					temp = 0.f;
					for (l = 0; l < k; l++)
						temp += a[l + i * lda] * b[j + l * ldb];
					if (beta == 0.f)
						c[i + j * ldc] = alpha * temp;
					else
						c[i + j * ldc] = alpha * temp + beta * c[i + j * ldc];
				}
			}
		}
	}
} /* End of SGEMM . */

void printVect(int n, const float * vect, const char * msg) {
	int i;
	if(msg != NULL) puts(msg);
	for(i = 0; i < n; i++) {
		printf("%6.4f, ", vect[i]);
		if((i+1)%10 == 0) printf("\n");
	}
	if(n%10 != 0) printf("\n");
	if(msg != NULL) puts("----------");
}

void printMat(int rows, int cols, const float * mat, const char * msg) {
	int i;
	if(msg != NULL) puts(msg);
	for(i = 0; i < rows; i++)
		printVect(cols, mat+i*cols, NULL);
	if(msg != NULL) puts("----------");
}

/*    A is (n*p) and B is (q*r), A*B returned to C  */
void mmult_JM (float *A, int n, int p, float *B, int q, int r, float *C) {
	if (p != q)
		error("Error, matrices not suitable\nfor multiplication");
	else {
		float
			* gpuA, * gpuB, * gpuC;

		cublasAlloc(n*p, sizeof(float), (void **) &gpuA);
		cublasAlloc(q*r, sizeof(float), (void **) &gpuB);
		cublasAlloc(n*r, sizeof(float), (void **) &gpuC);

		cublasSetMatrix(p, n, sizeof(float), A, p, gpuA, p);
		cublasSetMatrix(r, q, sizeof(float), B, r, gpuB, r);

		cublasSgemm('N', 'N', r, n, p, 1.f, gpuB, r, gpuA, p, 0.f, gpuC, r);

		cublasFree(gpuA);
		cublasFree(gpuB);

		cublasGetMatrix(r, n, sizeof(float), gpuC, r, C, r);
		cublasFree(gpuC);
		// printMat(n, r, C, "another C");
	}
}

void
orthog_mat_JM (float *mat, int e, float *orthog)
{
	/* take Wmat, (e*e), and return orthogonalized version to orthog_W */
	float *u, *v, *d, *temp;
	int i;

	u = Calloc (e * e, float);
	d = Calloc (e, float);
	v = Calloc (e * e, float);
	temp = Calloc (e * e, float);

	svd_JM (mat, &e, &e, u, d, v);
	for (i = 0; i < e; i++)
		temp[i * e + i] = 1 / (d[i]);

	mmult_JM (u, e, e, temp, e, e, v);
	transpose_mat_JM (u, &e, &e, temp);
	mmult_JM (v, e, e, temp, e, e, u);
	mmult_JM (u, e, e, mat, e, e, orthog);

	Free (u);
	Free (v);
	Free (d);
	Free (temp);
}

void
Symm_logcosh_JM (float *w_init, int e, float *data, int f, int p, float alpha, float *w_final, float *Tol)
{ /* Function that carries out Symmetric ICA using a logcosh approximation to the neg. entropy function */

	float *mat1, *mat2, *mat3, *mat4, *mat5, *mat6;
	int i, j;
	float mean;
	
	if (e != f) {
		error ("error in Symm_logcosh_JM, dims dont match");
	}
	else {
		mat1 = Calloc (e * p, float);
		mat2 = Calloc (e * p, float);
		mat3 = Calloc (e * e, float);
		mat4 = Calloc (e * e, float);
		mat5 = Calloc (e * e, float);
		mat6 = Calloc (e * e, float);
		
		mmult_JM (w_init, e, e, data, e, p, mat1);  
		
		
		for (i = 0; i < e; i++) {
			for (j = 0; j < p; j++) {
				mat1[i * p + j] = tanh (alpha * mat1[i * p + j]);
			}
		}			
		transpose_mat_JM (data, &e, &p, mat2);
		for (i = 0; i < e; i++) {
			for (j = 0; j < p; j++) {
				mat2[i * p + j] = (mat2[i * p + j]) / p;
			}
		}			
		mmult_JM (mat1, e, p, mat2, p, e, mat3);       
		for (i = 0; i < e; i++) {
			for (j = 0; j < p; j++) {
				mat1[i * p + j] =
					(alpha * (1 - (mat1[i * p + j]) * (mat1[i * p + j])));
			}
		}
		
		for (i = 0; i < e; i++) {
			mean = 0;
			for (j = 0; j < p; j++) {
				mean += ((mat1[i * p + j]) / p);
			}
			mat4[i * e + i] = mean;
		}		       
		mmult_JM (mat4, e, e, w_init, e, e, mat5); 
		for (i = 0; i < e; i++) {
			for (j = 0; j < e; j++) {
				mat4[i * e + j] = (mat3[i * e + j] - mat5[i * e + j]);
			}
		}
		
		transpose_mat_JM (w_init, &e, &e, mat6);
		orthog_mat_JM (mat4, e, w_final);
		
		
		mmult_JM (w_final, e, e, mat6, e, e, mat5);       
		mean = 0;
		for (i = 0; i < e; i++) {
			if (fabs (1 - fabs (mat5[i * e + i])) > mean) {
				mean = (fabs (1 - fabs (mat5[i * e + i])));
			}
		}
		*Tol = mean;
		Free (mat1);
		Free (mat2);
		Free (mat3);
		Free (mat4);
		Free (mat5);
		Free (mat6);
	}
}

void
Def_logcosh_JM (float *w_init, int e, float *data, int f, int p, float alpha, float *w_final)
{	
/* Function that carries out Deflation ICA using an logcosh approximation to the neg. entropy function */
	
	float *mat1, *mat2, *mat3, *mat4;
	int i, j;
	float mean;
	
	if (e != f) {
		error ("error in Def_logcosh_JM, dims dont match");
	}
	else {
		mat1 = Calloc (1 * p, float);
		mat2 = Calloc (e * p, float);
		mat3 = Calloc (1 * e, float);
		mat4 = Calloc (1 * e, float);
		
		mmult_JM (w_init, 1, e, data, e, p, mat1);
		
		
		for (i = 0; i < p; i++) {
			mat1[i] = tanh (alpha * mat1[i]);
		}			
		transpose_mat_JM (data, &e, &p, mat2);
		for (i = 0; i < e; i++) {
			for (j = 0; j < p; j++) {
				mat2[i * p + j] = (mat2[i * p + j]) / p;
			}
		}
		
		mmult_JM (mat1, 1, p, mat2, p, e, mat3);
		for (i = 0; i < p; i++) {
			mat1[i] = (alpha * (1 - (mat1[i]) * (mat1[i])));
		}
		
		mean = 0;
		for (j = 0; j < p; j++) {
			mean += ((mat1[j]) / p);
		}
		for (i = 0; i < e; i++) {
			mat4[i] = (w_init[i]) * mean;
		}			
		for (i = 0; i < e; i++) {
			w_final[i] = (mat3[i] - mat4[i]);
		}		
				
		Free (mat1);
		Free (mat2);
		Free (mat3);
		Free (mat4);
		
	}
}
	
	void
Symm_exp_JM (float *w_init, int e, float *data, int f, int p, float alpha, float *w_final, float *Tol)
		{	
    /* Function that carries out Symmetric ICA using a exponential approximation to the neg. entropy function */

float *mat1, *mat2, *mat3, *mat4, *mat5, *mat0, *mat6;
int i, j;
float mean;

if (e != f) {
    error ("error in Symm_exp_JM, dims dont match");
}
else {
    mat0 = Calloc (e * p, float);
    mat1 = Calloc (e * p, float);
    mat2 = Calloc (e * p, float);
    mat3 = Calloc (e * e, float);
    mat4 = Calloc (e * e, float);
    mat5 = Calloc (e * e, float);
    mat6 = Calloc (e * e, float);
    mmult_JM (w_init, e, e, data, e, p, mat1);  
    for (i = 0; i < e; i++) {
	for (j = 0; j < p; j++) {
	    mat0[i * p + j] =
		(mat1[i * p + j]) * exp (-0.5 * (mat1[i * p + j]) *
					 (mat1[i * p + j]));
	}
    }		      
    transpose_mat_JM (data, &e, &p, mat2);
    for (i = 0; i < e; i++) {
	for (j = 0; j < p; j++) {
	    mat2[i * p + j] = (mat2[i * p + j]) / p;
	}
    }		       
    mmult_JM (mat0, e, p, mat2, p, e, mat3);       
    for (i = 0; i < e; i++) {
	for (j = 0; j < p; j++) {
	    mat1[i * p + j] =
		((1 - (mat1[i * p + j]) * (mat1[i * p + j])) * 
		 exp (-0.5 * (mat1 [i * p + j]) * (mat1 [i * p + j])));
	}
    }

    for (i = 0; i < e; i++) {
	mean = 0;
	for (j = 0; j < p; j++) {
	    mean += ((mat1[i * p + j]) / p);
	}
	mat4[i * e + i] = mean;
    }		       
    mmult_JM (mat4, e, e, w_init, e, e, mat5); 
    for (i = 0; i < e; i++) {
	for (j = 0; j < e; j++) {
	    mat4[i * e + j] = (mat3[i * e + j] - mat5[i * e + j]);
	}
    }

    transpose_mat_JM (w_init, &e, &e, mat6);
    orthog_mat_JM (mat4, e, w_final);

    mmult_JM (w_final, e, e, mat6, e, e, mat5);	
    mean = 0;
    for (i = 0; i < e; i++) {
	if (fabs (1 - fabs (mat5[i * e + i])) > mean) {
	    mean = (fabs (1 - fabs (mat5[i * e + i])));
	}
    }
    *Tol = mean;
    Free (mat1);
    Free (mat2);
    Free (mat3);
    Free (mat4);
    Free (mat5);
    Free (mat0);
    Free (mat6);
}
}

void
Def_exp_JM (float *w_init, int e, float *data, int f, int p, float alpha, float *w_final)
{
    /* Function that carries out Deflation ICA using an exponential approximation to the neg. entropy function */

float *mat1, *mat2, *mat3, *mat4;
int i, j;
float mean;

if (e != f) {
    error ("error in Def_exp_JM, dims dont match");
}
else {
    mat1 = Calloc (1 * p, float);
    mat2 = Calloc (e * p, float);
    mat3 = Calloc (1 * e, float);
    mat4 = Calloc (1 * e, float);

    mmult_JM (w_init, 1, e, data, e, p, mat1);	

    for (i = 0; i < p; i++) {
	mat1[i] = ((mat1[i]) * exp (-0.5 * (mat1[i]) * (mat1[i])));
    }

    transpose_mat_JM (data, &e, &p, mat2);
    for (i = 0; i < e; i++) {
	for (j = 0; j < p; j++) {
	    mat2[i * p + j] = (mat2[i * p + j]) / p;
	}
    }

    mmult_JM (mat1, 1, p, mat2, p, e, mat3);

    mmult_JM (w_init, 1, e, data, e, p, mat1);
    for (i = 0; i < p; i++) {
	mat1[i] =
	    ((1 -
	      (mat1[i]) * (mat1[i])) * exp (-.5 * (mat1[i]) * (mat1[i])));
    }
    mean = 0;
    for (j = 0; j < p; j++) {
	mean += ((mat1[j]) / p);
    }
    for (i = 0; i < e; i++) {
	mat4[i] = (w_init[i]) * mean;
    }		     
    for (i = 0; i < e; i++) {
	w_final[i] = (mat3[i] - mat4[i]);
    }		       


    Free (mat1);
    Free (mat2);
    Free (mat3);
    Free (mat4);

}
}

void
gramsch_JM (float *ww, int n, int m, int k)
{
int ip, jp;
float tmp;
/* do Gram-Schmidt on row k of (n*m) matrix ww */
k -= 1;
if (k > n) {
    error ("Error in gramsch");
}
else {
    for (ip = 0; ip < k; ip++) {
	tmp = 0;
	for (jp = 0; jp < m; jp++) {
	    tmp += ((ww[m * ip + jp]) * (ww[m * k + jp]));
	}
	for (jp = 0; jp < m; jp++) {
	    ww[m * k + jp] = (ww[m * k + jp] - ((ww[m * ip + jp]) * tmp));
	}
    }
}
}

void
rowstd_JM (float *ww, int n, int m, int k)
{
/* for ww (n*m), make ||ww[k, ]|| equal 1 */
float tmp = 0;
int i;
k -= 1;
if (k > n) {
    error ("Error in rowstd");
}
else {
    for (i = 0; i < m; i++) {
	tmp += ((ww[k * m + i]) * (ww[k * m + i]));
    }
    tmp = sqrt (tmp);
    for (i = 0; i < m; i++) {
	ww[k * m + i] = ((ww[k * m + i]) / tmp);
    }
}
}


void 
calc_K_JM(float *x, int *n, int *p, float *K)
{
    int i, j;
    float *xxt, *xt, *u, *d, *v, *temp1, *temp2;

    xxt = Calloc (*n * *n, float);
    xt = Calloc (*n * *p, float);

    /* transpose x matrix */
    transpose_mat_JM (x, n, p, xt); 

    /* calculate sample covariance matrix xxt */
    mmult_JM (x, *n, *p, xt, *p, *n, xxt); 
    for (i = 0; i < *n; i++) {
	    for (j = 0; j < *n; j++) {
		    xxt[*n * i + j] = xxt[*n * i + j] / *p;
	    }
    }	
    Free (xt);

    /* calculate svd decomposition of xxt */ 
    u = Calloc (*n * *n, float);
    d = Calloc (*n, float);
    v = Calloc (*n * *n, float);

    svd_JM (xxt, n, n, u, d, v); 


    /* calculate K matrix*/
    temp1 = Calloc (*n * *n, float);
    temp2 = Calloc (*n * *n, float);

    for (i = 0; i < *n; i++) {
	    temp1[*n * i + i] = 1 / sqrt (d[i]);
    }

    transpose_mat_JM (u, n, n, temp2);
    mmult_JM (temp1, *n, *n, temp2, *n, *n, K);

    Free (temp1);
    Free (temp2);
    Free(xxt);
    Free(u);
    Free(d);
    Free(v);

}

void
calc_A_JM(float *w, float *k, float *data, int *e, int *n, int *p, float *A, float *unmixed_data)
{
	/* calculate un-mixing matrix A */
	int i;
	float *um, *umt, *umumt, *uu, *dd, *vv, *temp1, *temp2, *temp3;

	um = Calloc (*e * *n, float);
	umt = Calloc (*n * *e, float);
	
	mmult_JM (w, *e, *e, k, *e, *n, um);
	mmult_JM (um, *e, *n, data, *n, *p, unmixed_data);	
	transpose_mat_JM (um, e, n, umt);	
	
	umumt = Calloc (*e * *e, float);
	mmult_JM (um, *e, *n, umt, *n, *e, umumt);	
	
	uu = Calloc (*e * *e, float);
	dd = Calloc (*e, float);
	vv = Calloc (*e * *e, float);
	svd_JM (umumt, e, e, uu, dd, vv);
	
	temp1 = Calloc (*e * *e, float);
	for (i = 0; i < *e; i++) {
		temp1[*e * i + i] = 1 / (dd[i]);
	}
	
	temp2 = Calloc (*e * *e, float);
	temp3 = Calloc (*e * *e, float);
	transpose_mat_JM (vv, e, e, temp3);
	mmult_JM (temp3, *e, *e, temp1, *e, *e, temp2);
	transpose_mat_JM (uu, e, e, vv);
	mmult_JM (temp2, *e, *e, vv, *e, *e, uu);
	
	mmult_JM (umt, *n, *e, uu, *e, *e, A);

	Free(um);
	Free(umt);
	Free(umumt);
	Free(uu);
	Free(dd);
	Free(vv);
	Free(temp1);
	Free(temp2);
	Free(temp3);

}

void
icainc_JM (float *data_matrix, float *w_matrix, int *nn, int *pp, int *ee,
	float *alpha, int *rowflag, int *colflag, int *funflag, int *maxit,
	float *lim, int *defflag, int *verbose, float *data_pre, float *Kmat1,
	float *w_final, float *ansa, float *ansx2)
{

	/* main ICA function */
	
	int i, j, k, n, p, e;
	float tol;
	float *temp_w1, *temp_w2;
	float *data1, *Kmat, *temp1, *w_init;
	
	culaInitialize();	
	cublasInit();
	
	n = *nn;
	p = *pp;
	e = *ee;
	
	/* make a copy of the data matrix*/
	data1 = Calloc (n * p, float);
	for (i = 0; i < n; i++) {
		for (j = 0; j < p; j++) {
			data_pre[i * p + j] = data_matrix[i * p + j];
		}
	}
	
	/* row center data matrix if required*/
	if (*rowflag == 1) {
		rowcentre_JM (data_pre, n, p);
		if (*verbose == 1)
			Rprintf ("Centering\n");
	}
	
	/* standardize columns of data matrix if required*/
	if (*colflag == 1) {
		colstandard_JM (data_pre, n, p);	
		Rprintf("colstandard\n");
	} 
	
	/* calculate pre-whitening matrix Kmat */
	if (*verbose == 1)	Rprintf ("Whitening\n");
	Kmat = Calloc (n * n, float);    
	calc_K_JM(data_pre, &n, &p, Kmat); 
	
	/* pre-whiten data and reduce dimension from size n to size e */
       
	for (i = 0; i < e; i++) {
		for (j = 0; j < n; j++) {
			Kmat1[i * n + j] = Kmat[i * n + j];
		}
	}
	mmult_JM (Kmat1, e, n, data_pre, n, p, data1);
	
	/* calculate initial (orthogonal) unmixing matrix w */
	temp1 = Calloc (e * e, float);	
	w_init = Calloc (e * e, float);
	for (i = 0; i < e; i++) {
		for (j = 0; j < e; j++) {
			temp1[i * e + j] = w_matrix[i * e + j];
		}
	}
	orthog_mat_JM (temp1, e, w_init);     
	
	
	
	
	/* Main ICA code */
	
	
    if (*defflag == 0) {
	    if (*funflag == 1) {
		    
		    if (*verbose == 1)
			    Rprintf("Symmetric FastICA using logcosh approx. to neg-entropy function\n");
		    
		    i = 1;
		    Symm_logcosh_JM (w_init, e, data1, e, p, *alpha, w_final, &tol);
		    if (*verbose == 1)
			    Rprintf ("Iteration %d tol=%f\n", i, tol);
		    i = 2;
		    
		    while ((tol > (*lim)) && (i < (*maxit))) {
			    Symm_logcosh_JM (w_final, e, data1, e, p, *alpha, w_final, &tol);
			    if (*verbose == 1)
				    Rprintf ("Iteration %d tol=%f\n", i, tol);
			    i += 1;
	    }
	    }
	    
	    if (*funflag == 2) {
		    if (*verbose == 1)
			    Rprintf("Symmetric FastICA using exponential approx. to neg-entropy function\n");
		    
		    i = 1;
		    Symm_exp_JM (w_init, e, data1, e, p, *alpha, w_final, &tol);
		    if (*verbose == 1) Rprintf ("Iteration %d tol=%f\n", i, tol);
		    
		    i = 2;
		    while ((tol > (*lim)) && (i < (*maxit))) {
			    Symm_exp_JM (w_final, e, data1, e, p, *alpha, w_final, &tol);
			    if (*verbose == 1) Rprintf ("Iteration %d tol=%f\n", i, tol);
			    i += 1;
		    }
	    }
    }
    
    if (*defflag == 1) {
	    temp_w1 = Calloc (e, float);
	    temp_w2 = Calloc (e, float);
	    
	    if (*funflag == 1) {
		    if (*verbose == 1)
			    Rprintf ("Deflation FastICA using logcosh approx. to neg-entropy function\n");
		    
		    for (i = 0; i < e; i++) {
			    k = 0;
			    gramsch_JM (w_init, e, e, i + 1); 
			    rowstd_JM (w_init, e, e, i + 1);
			    tol = 1;
			    
			    while ((tol > (*lim)) && (k < (*maxit))) {
				    for (j = 0; j < e; j++) {
					    temp_w1[j] = w_init[i * e + j];
				    }
				    Def_logcosh_JM (temp_w1, e, data1, e, p, *alpha, temp_w2);
		    for (j = 0; j < e; j++) {
			    w_init[i * e + j] = temp_w2[j];
		    }
		    gramsch_JM (w_init, e, e, i + 1);
		    rowstd_JM (w_init, e, e, i + 1);
		    tol = 0;
		    for (j = 0; j < e; j++) {
			    tol += ((temp_w1[j]) * (w_init[i * e + j]));
		    }
		    tol = (fabs (fabs (tol) - 1));
		    k += 1;
			    }

			    if (*verbose == 1)
				    Rprintf ("Component %d needed %d iterations tol=%f\n",
					     i + 1, k, tol);
			    
		    }
	    }
	    if (*funflag == 2) {
		    
		    if (*verbose == 1)
			    Rprintf ("Deflation FastICA using exponential approx. to neg-entropy function\n");
		    
		    for (i = 0; i < e; i++) {
			    k = 0;
			    gramsch_JM (w_init, e, e, i + 1);
			    rowstd_JM (w_init, e, e, i + 1);
			    tol = 1;
			    
			    while ((tol > (*lim)) && (k < (*maxit))) {
				    for (j = 0; j < e; j++) {
					    temp_w1[j] = w_init[i * e + j];
		    }
				    Def_exp_JM (temp_w1, e, data1, e, p, *alpha, temp_w2);
				    for (j = 0; j < e; j++) {
					    w_init[i * e + j] = temp_w2[j];
				    }
				    gramsch_JM (w_init, e, e, i + 1);
				    rowstd_JM (w_init, e, e, i + 1);
				    tol = 0;
				    for (j = 0; j < e; j++) {
					    tol += ((temp_w1[j]) * (w_init[i * e + j]));
				    }
		    tol = (fabs (fabs (tol) - 1));
		    k += 1;
			    }

			    if (*verbose == 1)
				    Rprintf ("Component %d needed %d iterations tol=%f\n",
					     i + 1, k, tol);

		    }
	    }
	    for (i = 0; i < e; i++) {
		    for (j = 0; j < e; j++) {
			    w_final[i * e + j] = w_init[i * e + j];
		    }
	    }
	    Free (temp_w1);
	    Free (temp_w2);
    }

    /* calculate mixing matrix ansa */
    calc_A_JM(w_final, Kmat1, data_pre, &e, &n, &p, ansa, ansx2);

	culaShutdown();
    Free (data1);
    Free (Kmat);
    Free (temp1);
    Free (w_init);
}

#include <R_ext/Rdynload.h>

static const R_CMethodDef CEntries[] = {
    {"icainc_JM", (DL_FUNC) &icainc_JM, 18},
   {NULL, NULL, 0}
};


void
R_init_fastICA(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
