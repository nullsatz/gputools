/*
 * This svm, svr, and prediciton code is adapted from 
 * AUSTIN CARPENTER's cuSVM
 * found at
 * http://patternsonascreen.net/cuSVM.html
 * adapted by Josh Buckner
 * for use from within R using foreign function call 
 * support.  The rest is written by Josh Buckner.
 * Any bugs should be reported to Josh Buckner.
 */

#include<stdlib.h>
#include<assert.h>
#include<string.h>
#include<stdio.h>
#include<limits.h>
#include<ctype.h>
#include<float.h>
#include<vector>
#include<algorithm>
#include<math.h>
#include<cublas.h>
#include<cuda.h>
#include<R.h>

#include<cuseful.h>
#include<sort.h>
#include<classification.h>

#define MBtoLeave			 200

#define CUBIC_ROOT_MAX_OPS	2000

#define SAXPY_CTAS_MAX		  80
#define SAXPY_THREAD_MIN	  32
#define SAXPY_THREAD_MAX	 128
#define TRANS_BLOCK_DIM		  16

#define mxCUDA_SAFE_CALL(junk)	junk

using namespace std;

void VectorSplay (int n, int tMin, int tMax, int gridW, int *nbrCtas,
                        int *elemsPerCta, int *threadsPerCta)
{
    if (n < tMin) {
        *nbrCtas = 1;
        *elemsPerCta = n;
        *threadsPerCta = tMin;
    } else if (n < (gridW * tMin)) {
        *nbrCtas = ((n + tMin - 1) / tMin);
        *threadsPerCta = tMin;
        *elemsPerCta = *threadsPerCta;
    } else if (n < (gridW * tMax)) {
        int grp;
        *nbrCtas = gridW;
        grp = ((n + tMin - 1) / tMin);
        *threadsPerCta = (((grp + gridW -1) / gridW) * tMin);
        *elemsPerCta = *threadsPerCta;
    } else {
        int grp;
        *nbrCtas = gridW;
        *threadsPerCta = tMax;
        grp = ((n + tMin - 1) / tMin);
        grp = ((grp + gridW - 1) / gridW);
        *elemsPerCta = grp * tMin;
    }
}

__global__ void transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM+1];

	// read the matrix tile into shared memory
	unsigned int
		xIndex = blockIdx.x * TRANS_BLOCK_DIM + threadIdx.x,
		yIndex = blockIdx.y * TRANS_BLOCK_DIM + threadIdx.y;

	if((xIndex < width) && (yIndex < height)) {
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}
	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * TRANS_BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * TRANS_BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width)) {
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

__constant__ float C;
__constant__ float taumin;
__constant__ float kernelwidth;

template <unsigned int blockSize>
__global__ void FindBJ(float *d_F, float* d_y,float* d_alpha,float* d_KernelCol,float *g_odata,int* g_index,float BIValue, unsigned int n)
{

    __shared__ float sdata[blockSize];
    __shared__ int ind[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid]=-FLT_MAX;
    ind[tid]=0;

    float temp;
    float globaltemp;

    float LocalCloseY;
    float LocalFarY;
    float maxtemp;
    float denomclose;
    float denomfar=1.f;


    while (i < n)
    {
        LocalCloseY=d_y[i];
        LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0.f;
        denomclose=(2.f-2.f*d_KernelCol[i]);
        if(i+blockSize<n){denomfar=(2.f-2.f*d_KernelCol[i+blockSize]);}


        denomclose=denomclose<taumin?taumin:denomclose;
        denomfar=denomfar<taumin?taumin:denomfar;


        maxtemp=
        fmaxf(
        globaltemp=
        (LocalCloseY*d_alpha[i])>(LocalCloseY==1?0:-C) ?
        __fdividef(__powf(BIValue+LocalCloseY*d_F[i],2.f),denomclose)
        :-FLT_MAX,
        i+blockSize<n ?
        ((LocalFarY*d_alpha[i+blockSize])>(LocalFarY==1?0:-C)?
        __fdividef(__powf(BIValue+LocalFarY*d_F[i+blockSize],2.f),denomfar)
        :-FLT_MAX)
        :-FLT_MAX);

        sdata[tid]=fmaxf(temp=sdata[tid],maxtemp);

        if (sdata[tid]!=temp)
        {
            sdata[tid]== globaltemp ? ind[tid]=i : ind[tid]=i+blockSize;
        }

    i += gridSize;
    }


    __syncthreads();

    if (tid < 128){ if (sdata[tid] < sdata[tid + 128]){ ind[tid]=ind[tid+128];sdata[tid]=sdata[tid+128];  }} __syncthreads();

    if (tid < 64){ if (sdata[tid] < sdata[tid + 64]){ ind[tid]=ind[tid+64];sdata[tid]=sdata[tid+64];  }} __syncthreads();

    if (tid < 32)
    {
        if (sdata[tid] <sdata[tid + 32]) {ind[tid]=ind[tid+32];sdata[tid]=sdata[tid+32];}
        if (sdata[tid] <sdata[tid + 16]) {ind[tid]=ind[tid+16];sdata[tid]=sdata[tid+16];}
        if (sdata[tid] <sdata[tid + 8]) {ind[tid]=ind[tid+8];sdata[tid]=sdata[tid+8];}
        if (sdata[tid] <sdata[tid + 4]) {ind[tid]=ind[tid+4];sdata[tid]=sdata[tid+4];}
        if (sdata[tid] <sdata[tid + 2]) {ind[tid]=ind[tid+2];sdata[tid]=sdata[tid+2];}
        if (sdata[tid] <sdata[tid + 1]) {ind[tid]=ind[tid+1];sdata[tid]=sdata[tid+1];}
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    if (tid == 0) g_index[blockIdx.x] = ind[0];
}

template <unsigned int blockSize>
__global__ void FindBI(float *d_F, float* d_y,float* d_alpha,float *g_odata,int* g_index,unsigned int n)
{

    __shared__ float sdata[blockSize];
    __shared__ int ind[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid]=-FLT_MAX;
    ind[tid]=0;



    float temp;
    float globaltemp;

    float LocalCloseY;
    float LocalFarY;
    float maxtemp;


    while (i < n)
    {
        LocalCloseY=d_y[i];
        LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0;

        maxtemp=
        fmaxf(
        globaltemp=
        (LocalCloseY*d_alpha[i])<(LocalCloseY==1?C:0) ?
        -(d_F[i]*LocalCloseY)
        :-FLT_MAX,
        i+blockSize<n ?
        ((LocalFarY*d_alpha[i+blockSize])<(LocalFarY==1?C:0) ?
        -(d_F[i+blockSize]*LocalFarY)
        :-FLT_MAX)
        :-FLT_MAX);

        sdata[tid]=fmaxf(temp=sdata[tid],maxtemp);

        if (sdata[tid]!=temp)
        {
            sdata[tid]== globaltemp ? ind[tid]=i : ind[tid]=i+blockSize;
        }

    i += gridSize;
    }


    __syncthreads();

    if (tid < 128){ if (sdata[tid] < sdata[tid + 128]){ ind[tid]=ind[tid+128];sdata[tid]=sdata[tid+128];  }} __syncthreads();

    if (tid < 64){ if (sdata[tid] < sdata[tid + 64]){ ind[tid]=ind[tid+64];sdata[tid]=sdata[tid+64];  }} __syncthreads();

    if (tid < 32)
    {
        if (sdata[tid] <sdata[tid + 32]) {ind[tid]=ind[tid+32];sdata[tid]=sdata[tid+32];}
        if (sdata[tid] <sdata[tid + 16]) {ind[tid]=ind[tid+16];sdata[tid]=sdata[tid+16];}
        if (sdata[tid] <sdata[tid + 8]) {ind[tid]=ind[tid+8];sdata[tid]=sdata[tid+8];}
        if (sdata[tid] <sdata[tid + 4]) {ind[tid]=ind[tid+4];sdata[tid]=sdata[tid+4];}
        if (sdata[tid] <sdata[tid + 2]) {ind[tid]=ind[tid+2];sdata[tid]=sdata[tid+2];}
        if (sdata[tid] <sdata[tid + 1]) {ind[tid]=ind[tid+1];sdata[tid]=sdata[tid+1];}

    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    if (tid == 0) g_index[blockIdx.x] = ind[0];
}






template <unsigned int blockSize>
__global__ void FindStoppingJ(float *d_F, float* d_y,float* d_alpha,float *g_odata,unsigned int n)
{

    __shared__ float sdata[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid]=FLT_MAX;


    float LocalCloseY;
    float LocalFarY;


    while (i < n)
    {
        LocalCloseY=d_y[i];
        LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0;

        sdata[tid]=
        fminf(
        sdata[tid],
        fminf(
        (LocalCloseY*d_alpha[i])>(LocalCloseY==1?0:-C) ?
        -(d_F[i]*LocalCloseY)
        :FLT_MAX,
        i+blockSize<n ?
        ((LocalFarY*d_alpha[i+blockSize])>(LocalFarY==1?0:-C)?
        -(d_F[i+blockSize]*LocalFarY)
        :FLT_MAX)
        :FLT_MAX));

    i += gridSize;
    }


    __syncthreads();

    if (tid < 128){ sdata[tid]=fminf(sdata[tid],sdata[tid+128]);} __syncthreads();

    if (tid < 64){ sdata[tid]=fminf(sdata[tid],sdata[tid+64]);} __syncthreads();

    if (tid < 32) {
        sdata[tid]=fminf(sdata[tid],sdata[tid+32]);
        sdata[tid]=fminf(sdata[tid],sdata[tid+16]);
        sdata[tid]=fminf(sdata[tid],sdata[tid+8]);
        sdata[tid]=fminf(sdata[tid],sdata[tid+4]);
        sdata[tid]=fminf(sdata[tid],sdata[tid+2]);
        sdata[tid]=fminf(sdata[tid],sdata[tid+1]);


    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}




__global__ void UpdateF(float * F,float *KernelColI,float* KernelColJ, float* d_y,float deltaalphai,float deltaalphaj,float yi,float yj,int n)
{

    int totalThreads,ctaStart,tid;
    totalThreads = gridDim.x*blockDim.x;
    ctaStart = blockDim.x*blockIdx.x;
    tid = threadIdx.x;
    int i;

    for (i = ctaStart + tid; i < n; i += totalThreads)
            {
                F[i] = F[i] + yi*d_y[i]*deltaalphai*KernelColI[i]+yj*d_y[i]*deltaalphaj*KernelColJ[i];
            }


}

__global__ void RBFFinish(float *KernelCol, const float * KernelDotProd,const float* DotProd,const float* DotProdRow,const int n)
{

    int totalThreads,ctaStart,tid;
    totalThreads = gridDim.x*blockDim.x;
    ctaStart = blockDim.x*blockIdx.x;
    tid = threadIdx.x;
    int i;

    for (i = ctaStart + tid; i < n; i += totalThreads)
        KernelCol[i] = expf(kernelwidth*(DotProd[i]+*DotProdRow-KernelDotProd[i]*2.f));
}






inline void RBFKernel(float *d_KernelJ,const int BJIndex,const float *d_x,const float * d_Kernel_InterRow,float *d_KernelDotProd, float *d_SelfDotProd,const int& m,const int& n,const int &nbrCtas,const int& threadsPerCta)
{

    cublasSgemv ('n', m, n, 1,d_x, m, d_Kernel_InterRow, 1, 0, d_KernelDotProd, 1);

    RBFFinish<<<nbrCtas,threadsPerCta>>>(d_KernelJ, d_KernelDotProd,d_SelfDotProd,d_SelfDotProd+BJIndex,m);

}



inline void CpuMaxInd(float &BIValue, int &BIIndex,const float * value_inter,const  int * index_inter,const  int n)
{

    BIValue=value_inter[0];
    BIIndex=index_inter[0];

    for(int j=0;j<n;j++)
    {
        if (value_inter[j]>BIValue)
        {
        BIValue=value_inter[j];
        BIIndex=index_inter[j];

        }
    }

}




inline void CpuMaxIndSvr(float &BIValue, int &BIIndex, const  float * value_inter,const  int * index_inter,int n,const  int m)
{

    BIValue=value_inter[0];
    BIIndex=index_inter[0];

    for(int j=0;j<n;j++)
    {
        if (value_inter[j]>BIValue)
        {
        BIValue=value_inter[j];
        BIIndex=j<n/2?index_inter[j]:index_inter[j]+m;

        }
    }

}




inline void CpuMin(float &SJValue, float * value_inter,int n)
{

    SJValue=value_inter[0];

    for(int j=0;j<n;j++)
    {
        if (value_inter[j]<SJValue)
        {
            SJValue=value_inter[j];

        }
    }

}



inline void DotProdVector(float * x, float* dotprod,int m, int n)
{

    for(int i=0;i<m;i++)
    {
        dotprod[i]=0;

        for(int j=0;j<n;j++)
            dotprod[i]+=(x[i+j*m])*(x[i+j*m]);

    }



}

inline void IncrementKernelCache(vector<int>& KernelCacheItersSinceUsed,const int &RowsInKernelCache)
 {
    for(int k=0;k<RowsInKernelCache;k++)
    {
        KernelCacheItersSinceUsed[k]+=1;
    }
}









inline void UpdateAlphas(float& alphai,float& alphaj,const float& Kij,const float& yi,const float& yj,const float& Fi,const float& Fj,const float& C,const float& h_taumin)
{

//This alpha update code is adapted from that in LIBSVM.
//Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines, 2001. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

    float lambda;
    float lambda_denom;


    lambda_denom=2.0-2.0*Kij;
    if (lambda_denom<h_taumin) {lambda_denom=h_taumin;}

    if (yi!=yj)
    {
    lambda=(-Fi-Fj)/lambda_denom;
    float alphadiff=alphai-alphaj;

    alphai+=lambda;
    alphaj+=lambda;


		    if(alphadiff > 0)
			    {
				    if(alphaj < 0)
				    {
					    alphaj = 0;
					    alphai = alphadiff;
				    }



			    }
			    else
			    {
				    if(alphai < 0)
				    {
					    alphai = 0;
					    alphaj = -alphadiff;
				    }
			    }


                 if(alphadiff > 0)
			    {
				    if(alphai > C)
				    {
					    alphai = C;
					    alphaj = C - alphadiff;
				    }
			    }
			    else
			    {
				    if(alphaj > C)
				    {
					    alphaj = C;
					    alphai = C + alphadiff;
				    }
			    }


    }
    else
    {
    float alphasum=alphai+alphaj;
    lambda=(Fi-Fj)/lambda_denom;
    alphai-=lambda;
    alphaj+=lambda;

    	    if(alphasum > C)
			    {
				    if(alphai > C)
				    {
					    alphai = C;
					    alphaj = alphasum - C;
				    }
                    if(alphaj > C)
				    {
					    alphaj = C;
					    alphai = alphasum - C;
				    }
			    }
			    else
			    {
				    if(alphaj < 0)
				    {
					    alphaj = 0;
					    alphai = alphasum;
				    }
	                if(alphai < 0)
				    {
					    alphai = 0;
					    alphaj = alphasum;
				    }
			    }

    }

}

void getSupportVectors(int isRegression, int m, int n, int numSVs,
	int numPosSVs, const float * x, const float * y, const float * alphas,
	float * svCoefficients, float * supportVectors)
{
	if(!isRegression) {
		int
			PosSvIndex = 0, NegSvIndex = 0;

		for(int k = 0; k < m; k++) {
			if(alphas[k] != 0.f) {
				if(y[k] > 0.f) {
					svCoefficients[PosSvIndex] = alphas[k];
					for(int j = 0; j < n; j++)
						supportVectors[PosSvIndex+j*numSVs] = x[k+j*m];

					PosSvIndex++;
				} else {
					svCoefficients[NegSvIndex+numPosSVs] = alphas[k];
					for(int j = 0; j < n; j++) {
						supportVectors[NegSvIndex+numPosSVs+j*numSVs]
							= x[k+j*m];
					}
					NegSvIndex++;
				}
			}
		}
	} else {
		int svindex = 0;

		for(int k = 0; k < m; k++) {
			if(alphas[k] != 0.f) {
				svCoefficients[svindex] = alphas[k];
				for(int j = 0; j < n; j++)
						supportVectors[svindex+j*numSVs] = x[k+j*m];

				svindex++;
			}
		}
	}
}

/*
 * Outputs:
 * 1. alpha is a length m single-precision vector of the support vector
 *    coefficients.
 * 2. beta is a single-precision scalar, the offset b in the SVM prediction
 *    function.
 * 3. svs is a single-precision matrix of the support vectors corresponding to
 *    alphas, i.e. the support vector found in row i of svs has the
 *    coefficient in the SVM prediction function found in row i of alphas.
 *
 * Inputs:
 * 1. y is a single-precision vector of training outputs.  In regression,
 * 	  y's value may be continuously valued.  Otherwise, y's values must be
 *	  each from {-1, 1}.
 * 2. train is a single-precision matrix of training data corresponding to y.
 * 3. C is the scalar SVM regularization parameter.
 * 4. kernelWidth is the scalar Gaussian kernel parameter,
 *    i.e. lambda in exp(−lambda * norm(x − z)).
 * 5. eps is epsilon in epsilon-Support Vector Regression.
 * 6. StoppingCrit is an optional scalar argument that one can use to specify
 *    the optimization stopping criterion,  0.001f is recommended.
 */

void SVRTrain(float * mexalpha, float * beta, float * y, float * x ,float C,
	float kernelwidth, float eps, int m, int n, float StoppingCrit,
	int * numSvs)
{
/*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    cublasInit();

    int numBlocks=64;
    dim3 ReduceGrid(numBlocks, 1, 1);
    dim3 ReduceBlock(256, 1, 1);

    float h_taumin=0.0001;
    mxCUDA_SAFE_CALL(cudaMemcpyToSymbol("taumin", &h_taumin, sizeof(float)));

    kernelwidth*=-1;
    mxCUDA_SAFE_CALL(cudaMemcpyToSymbol("kernelwidth", &kernelwidth,
		sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMemcpyToSymbol("C", &C, sizeof(float)));

    float *alphasvr=new float [2*m];
    float *ybinary=new float [2*m];
    float *F=new float [2*m];

    for(int j=0;j<m;j++) {
        alphasvr[j]=0;
        ybinary[j]=1;
        F[j]=-y[j]+eps;

        alphasvr[j+m]=0;
        ybinary[j+m]=-1;
        F[j+m]=y[j]+eps;
    }

    float *SelfDotProd=new float [m];
    DotProdVector(x, SelfDotProd,m, n);

    int
		nbrCtas, elemsPerCta, threadsPerCta;

    VectorSplay(m, SAXPY_THREAD_MIN, SAXPY_THREAD_MAX, SAXPY_CTAS_MAX,
		&nbrCtas, &elemsPerCta,&threadsPerCta);

    float
		* d_x, * d_xT, * d_y,
    	* d_alpha,
    	* d_F,
    	* d_KernelDotProd, * d_SelfDotProd,
    	* d_KernelJ, * d_KernelI,
    	* d_KernelInterRow;

    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_x, m*n*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_xT, m*n*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMemcpy(d_x, x, sizeof(float)*n*m,
		cudaMemcpyHostToDevice));

	int
		numBlocksX, numBlocksY;

	numBlocksX = (int) ceil((float)m / TRANS_BLOCK_DIM);
	numBlocksY = (int) ceil((float)n / TRANS_BLOCK_DIM);

    dim3
		gridtranspose(numBlocksX, numBlocksY),
    	threadstranspose(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM, 1);

    cudaThreadSynchronize();
    transpose<<< gridtranspose, threadstranspose >>>(d_xT, d_x, m, n);

    float *xT=new float [n*m];
    mxCUDA_SAFE_CALL(cudaMemcpy(xT, d_xT, sizeof(float)*m*n,cudaMemcpyDeviceToHost));
    mxCUDA_SAFE_CALL(cudaFree(d_xT));


    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_KernelInterRow, n*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_alpha, 2*m*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_y, 2*m*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_F, 2*m*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_SelfDotProd, m*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_KernelDotProd, m*sizeof(float)));




    mxCUDA_SAFE_CALL(cudaMemcpy(d_y, ybinary, sizeof(float)*m*2,cudaMemcpyHostToDevice));
    mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha, alphasvr, sizeof(float)*m*2,cudaMemcpyHostToDevice));
    mxCUDA_SAFE_CALL(cudaMemcpy(d_F, F, sizeof(float)*m*2,cudaMemcpyHostToDevice));
    mxCUDA_SAFE_CALL(cudaMemcpy(d_SelfDotProd, SelfDotProd, sizeof(float)*m,cudaMemcpyHostToDevice));



    delete [] F;
    delete [] SelfDotProd;


    float* value_inter;
    int* index_inter;
    float* value_inter_svr;
    int* index_inter_svr;



    cudaMallocHost( (void**)&value_inter, numBlocks*sizeof(float) );
    cudaMallocHost( (void**)&index_inter, numBlocks*sizeof(int) );
    cudaMallocHost( (void**)&value_inter_svr, 2*numBlocks*sizeof(float) );
    cudaMallocHost( (void**)&index_inter_svr, 2*numBlocks*sizeof(int) );




    float* d_value_inter;
    int* d_index_inter;


    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_value_inter, numBlocks*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_index_inter, numBlocks*sizeof(int)));


    size_t free, total;
    cuMemGetInfo(&free, &total);


    int KernelCacheSize=free-MBtoLeave*1024*1024;
    int RowsInKernelCache=KernelCacheSize/(sizeof(float)*m);

    float *d_Kernel_Cache;
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_Kernel_Cache, KernelCacheSize));

    vector<int> KernelCacheIndices(RowsInKernelCache,-1);
    vector<int> KernelCacheItersSinceUsed(RowsInKernelCache,0);
    vector<int>::iterator CachePosI;
    vector<int>::iterator CachePosJ;
    int CacheDiffI;
    int CacheDiffJ;





    int CheckStoppingCritEvery=255;
    int iter=0;

    float BIValue;
    int BIIndex;
    float SJValue;
    float BJSecondOrderValue;
    int BJIndex;
    float Kij;
    float yj;
    float yi;
    float alphai;
    float alphaj;
    float oldalphai;
    float oldalphaj;
    float Fi;
    float Fj;


        while (1)
        {



            FindBI<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_value_inter,d_index_inter, 2*m);

            mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
            mxCUDA_SAFE_CALL(cudaMemcpy(index_inter, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));
            cudaThreadSynchronize();
            CpuMaxInd(BIValue,BIIndex,value_inter,index_inter,numBlocks);

            if ((iter & CheckStoppingCritEvery)==0)
            {
                FindStoppingJ<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_value_inter, 2*m);

                mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
                cudaThreadSynchronize();
                CpuMin(SJValue,value_inter,numBlocks);

                if(BIValue-SJValue<StoppingCrit) {*beta=(SJValue+BIValue)/2; break;}
            }


            CachePosI=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),(BIIndex>=m?BIIndex-m:BIIndex));
            if (CachePosI ==KernelCacheIndices.end())
            {
                CacheDiffI=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
                d_KernelI=d_Kernel_Cache+CacheDiffI*m;
                mxCUDA_SAFE_CALL(cudaMemcpy(d_KernelInterRow, xT+(BIIndex>=m?BIIndex-m:BIIndex)*n, n*sizeof(float),cudaMemcpyHostToDevice));
                RBFKernel(d_KernelI,(BIIndex>=m?BIIndex-m:BIIndex),d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd,m,n,nbrCtas,threadsPerCta);

                *(KernelCacheIndices.begin()+CacheDiffI)=(BIIndex>=m?BIIndex-m:BIIndex);
            }
            else
            {
                CacheDiffI=CachePosI-KernelCacheIndices.begin();
                d_KernelI=d_Kernel_Cache+m*CacheDiffI;
            }
            *(KernelCacheItersSinceUsed.begin()+CacheDiffI)=-1;




            FindBJ<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_KernelI,d_value_inter,d_index_inter,BIValue, m);

            mxCUDA_SAFE_CALL(cudaMemcpy(value_inter_svr, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
            mxCUDA_SAFE_CALL(cudaMemcpy(index_inter_svr, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));


            FindBJ<256><<<ReduceGrid, ReduceBlock>>>(d_F+m, d_y+m,d_alpha+m,d_KernelI,d_value_inter,d_index_inter,BIValue,m);

            mxCUDA_SAFE_CALL(cudaMemcpy(value_inter_svr+numBlocks, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
            mxCUDA_SAFE_CALL(cudaMemcpy(index_inter_svr+numBlocks, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));
            cudaThreadSynchronize();
            CpuMaxIndSvr(BJSecondOrderValue,BJIndex,value_inter_svr,index_inter_svr,2*numBlocks,m);


            mxCUDA_SAFE_CALL(cudaMemcpy(&Kij, d_KernelI+(BJIndex>=m?BJIndex-m:BJIndex), sizeof(float),cudaMemcpyDeviceToHost));

            mxCUDA_SAFE_CALL(cudaMemcpy(&alphai, d_alpha+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
            mxCUDA_SAFE_CALL(cudaMemcpy(&alphaj, d_alpha+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));


            mxCUDA_SAFE_CALL(cudaMemcpy(&yi, d_y+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
            mxCUDA_SAFE_CALL(cudaMemcpy(&yj, d_y+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));
            mxCUDA_SAFE_CALL(cudaMemcpy(&Fi, d_F+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
            mxCUDA_SAFE_CALL(cudaMemcpy(&Fj, d_F+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));

            oldalphai=alphai;
            oldalphaj=alphaj;


            UpdateAlphas(alphai,alphaj,Kij,yi,yj,Fi,Fj,C,h_taumin);



            mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha+BIIndex, &alphai, sizeof(float),cudaMemcpyHostToDevice));
            mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha+BJIndex, &alphaj, sizeof(float),cudaMemcpyHostToDevice));

            float deltaalphai = alphai - oldalphai;
            float deltaalphaj = alphaj - oldalphaj;




            CachePosJ=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),(BJIndex>=m?BJIndex-m:BJIndex));
            if (CachePosJ ==KernelCacheIndices.end())
            {
                CacheDiffJ=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
                d_KernelJ=d_Kernel_Cache+CacheDiffJ*m;
                mxCUDA_SAFE_CALL(cudaMemcpy(d_KernelInterRow, xT+(BJIndex>=m?BJIndex-m:BJIndex)*n, n*sizeof(float),cudaMemcpyHostToDevice));
                RBFKernel(d_KernelJ,(BJIndex>=m?BJIndex-m:BJIndex),d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd,m,n,nbrCtas,threadsPerCta);


                *(KernelCacheIndices.begin()+CacheDiffJ)=(BJIndex>=m?BJIndex-m:BJIndex);
            }
            else
            {
                CacheDiffJ=CachePosJ-KernelCacheIndices.begin();
                d_KernelJ=d_Kernel_Cache+m*CacheDiffJ;

            }


            UpdateF<<<nbrCtas,threadsPerCta>>>(d_F,d_KernelI,d_KernelJ,d_y,deltaalphai,deltaalphaj,yi,yj,m);

            UpdateF<<<nbrCtas,threadsPerCta>>>(d_F+m,d_KernelI,d_KernelJ,d_y+m,deltaalphai,deltaalphaj,yi,yj,m);


            IncrementKernelCache(KernelCacheItersSinceUsed,RowsInKernelCache);

            *(KernelCacheItersSinceUsed.begin()+CacheDiffI)=0;
            *(KernelCacheItersSinceUsed.begin()+CacheDiffJ)=0;



        iter++;

        }

    cublasGetVector(m*2,sizeof(float),d_alpha,1,alphasvr,1);

    for(int k=0;k<m;k++)
        mexalpha[k]=(alphasvr[k]-alphasvr[k+m])*ybinary[k];

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    delete [] ybinary;
    delete [] alphasvr;
    delete [] xT;

    cudaFreeHost(value_inter_svr);
    cudaFreeHost(index_inter_svr);
    cudaFreeHost(value_inter);
    cudaFreeHost(index_inter);

    mxCUDA_SAFE_CALL(cudaFree(d_x));
    mxCUDA_SAFE_CALL(cudaFree(d_y));
    mxCUDA_SAFE_CALL(cudaFree(d_alpha));
    mxCUDA_SAFE_CALL(cudaFree(d_Kernel_Cache));
    mxCUDA_SAFE_CALL(cudaFree(d_KernelInterRow));
    mxCUDA_SAFE_CALL(cudaFree(d_F));
    mxCUDA_SAFE_CALL(cudaFree(d_value_inter));
    mxCUDA_SAFE_CALL(cudaFree(d_index_inter));
    mxCUDA_SAFE_CALL(cudaFree(d_SelfDotProd));
    mxCUDA_SAFE_CALL(cudaFree(d_KernelDotProd));
    mxCUDA_SAFE_CALL( cudaThreadExit());

	*numSvs = 0;
	for(int k = 0; k < m; k++) {
		if(mexalpha[k])
			(*numSvs)++;
	}
*/
}

void SVMTrain(float *mexalpha,float* beta,float*y,float *x ,float C,
	float kernelwidth, int m, int n, float StoppingCrit,
	int * numSvs, int * numPosSvs)
{
/*    cudaEvent_t
		start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    int numBlocks = 64;
    dim3 ReduceGrid(numBlocks, 1, 1);
    dim3 ReduceBlock(256, 1, 1);

    float h_taumin = 0.0001;
    mxCUDA_SAFE_CALL(cudaMemcpyToSymbol("taumin", &h_taumin, sizeof(float)));

    kernelwidth*=-1;
    mxCUDA_SAFE_CALL(cudaMemcpyToSymbol("kernelwidth", &kernelwidth, sizeof(float)));

    mxCUDA_SAFE_CALL(cudaMemcpyToSymbol("C", &C, sizeof(float)));

    float
		* h_alpha = new float [m],
		* h_F = new float [m];

    for(int j = 0; j < m; j++) {
        h_alpha[j] = 0;
        h_F[j] = -1;
    }

    float *SelfDotProd=new float [m];
    DotProdVector(x, SelfDotProd,m, n);

    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;

    VectorSplay (m, SAXPY_THREAD_MIN, SAXPY_THREAD_MAX, SAXPY_CTAS_MAX, &nbrCtas, &elemsPerCta,&threadsPerCta);

    float * d_x;
    float * d_xT;
    float * d_alpha;
    float* d_y;
    float* d_F;
    float *d_KernelDotProd;
    float *d_SelfDotProd;
    float *d_KernelJ;
    float *d_KernelI;

    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_x, m*n*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_xT, m*n*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMemcpy(d_x, x, sizeof(float) * n * m,
		cudaMemcpyHostToDevice));

	int
		numBlocksX, numBlocksY;

	numBlocksX = (int) ceil((float)m / TRANS_BLOCK_DIM);
	numBlocksY = (int) ceil((float)n / TRANS_BLOCK_DIM);

    dim3
		gridtranspose(numBlocksX, numBlocksY),
    	threadstranspose(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM, 1);

    cudaThreadSynchronize();
    transpose<<< gridtranspose, threadstranspose >>>(d_xT, d_x, m, n);

    float *xT=new float [n*m];
    mxCUDA_SAFE_CALL(cudaMemcpy(xT, d_xT, sizeof(float)*m*n,cudaMemcpyDeviceToHost));
    mxCUDA_SAFE_CALL(cudaFree(d_xT));


    float* d_KernelInterRow;
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_KernelInterRow, n*sizeof(float)));


    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_alpha, m*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_y, m*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_F, m*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_SelfDotProd, m*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_KernelDotProd, m*sizeof(float)));

    mxCUDA_SAFE_CALL(cudaMemcpy(d_y, y, sizeof(float)*m,cudaMemcpyHostToDevice));
    mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha, h_alpha, sizeof(float)*m,cudaMemcpyHostToDevice));
    mxCUDA_SAFE_CALL(cudaMemcpy(d_F, h_F, sizeof(float)*m,cudaMemcpyHostToDevice));
    mxCUDA_SAFE_CALL(cudaMemcpy(d_SelfDotProd, SelfDotProd, sizeof(float)*m,cudaMemcpyHostToDevice));



    delete [] SelfDotProd;


    float* value_inter;
    int* index_inter;


    cudaMallocHost( (void**)&value_inter, numBlocks*sizeof(float) );
    cudaMallocHost( (void**)&index_inter, numBlocks*sizeof(int) );


    float* d_value_inter;
    int* d_index_inter;


    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_value_inter, numBlocks*sizeof(float)));
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_index_inter, numBlocks*sizeof(int)));

    size_t free, total;
    cuMemGetInfo(&free, &total);

    int KernelCacheSize=free-MBtoLeave*1024*1024;
    int RowsInKernelCache=KernelCacheSize/(sizeof(float)*m);


    float *d_Kernel_Cache;
    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_Kernel_Cache, KernelCacheSize));


    vector<int> KernelCacheIndices(RowsInKernelCache,-1);
    vector<int> KernelCacheItersSinceUsed(RowsInKernelCache,0);
    vector<int>::iterator CachePosI;
    vector<int>::iterator CachePosJ;
    int CacheDiffI;
    int CacheDiffJ;





    int CheckStoppingCritEvery=255;
    int iter=0;

    float BIValue;
    int BIIndex;
    float SJValue;
    float BJSecondOrderValue;
    int BJIndex;
    float Kij;
    float yj;
    float yi;
    float alphai;
    float alphaj;
    float oldalphai;
    float oldalphaj;
    float Fi;
    float Fj;

        while (1)
        {

        FindBI<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_value_inter,d_index_inter, m);
        mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
        mxCUDA_SAFE_CALL(cudaMemcpy(index_inter, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));
        cudaThreadSynchronize();
        CpuMaxInd(BIValue,BIIndex,value_inter,index_inter,numBlocks);

        cudaMemcpy(&Fi, d_F+BIIndex, sizeof(float),cudaMemcpyDeviceToHost);

        if ((iter & CheckStoppingCritEvery)==0)
        {
            FindStoppingJ<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_value_inter, m);
            mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
            cudaThreadSynchronize();
            CpuMin(SJValue,value_inter,numBlocks);

            if(BIValue-SJValue<StoppingCrit) {
            	if(BIValue-SJValue<StoppingCrit) {
            		*beta=(SJValue+BIValue)/2;
            		break;
            	}
            }
        }




        CachePosI=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),BIIndex);
        if (CachePosI ==KernelCacheIndices.end())
        {
            CacheDiffI=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
            d_KernelI=d_Kernel_Cache+CacheDiffI*m;
            mxCUDA_SAFE_CALL(cudaMemcpy(d_KernelInterRow, xT+BIIndex*n, n*sizeof(float),cudaMemcpyHostToDevice));
            RBFKernel(d_KernelI,BIIndex,d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd,m,n,nbrCtas,threadsPerCta);
            *(KernelCacheIndices.begin()+CacheDiffI)=BIIndex;
        }
        else
        {
            CacheDiffI=CachePosI-KernelCacheIndices.begin();
            d_KernelI=d_Kernel_Cache+m*CacheDiffI;
        }
        *(KernelCacheItersSinceUsed.begin()+CacheDiffI)=-1;




        FindBJ<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_KernelI,d_value_inter,d_index_inter,BIValue, m);
        mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
        mxCUDA_SAFE_CALL(cudaMemcpy(index_inter, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));
        cudaThreadSynchronize();
        CpuMaxInd(BJSecondOrderValue,BJIndex,value_inter,index_inter,numBlocks);


        mxCUDA_SAFE_CALL(cudaMemcpy(&Kij, d_KernelI+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));

        mxCUDA_SAFE_CALL(cudaMemcpy(&alphai, d_alpha+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
        mxCUDA_SAFE_CALL(cudaMemcpy(&alphaj, d_alpha+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));

        mxCUDA_SAFE_CALL(cudaMemcpy(&yi, d_y+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
        mxCUDA_SAFE_CALL(cudaMemcpy(&yj, d_y+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));
        mxCUDA_SAFE_CALL(cudaMemcpy(&Fj, d_F+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));


        oldalphai=alphai;
        oldalphaj=alphaj;


        UpdateAlphas(alphai,alphaj,Kij,yi,yj,Fi,Fj,C,h_taumin);



        mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha+BIIndex, &alphai, sizeof(float),cudaMemcpyHostToDevice));
        mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha+BJIndex, &alphaj, sizeof(float),cudaMemcpyHostToDevice));

        float deltaalphai = alphai - oldalphai;
        float deltaalphaj = alphaj - oldalphaj;




        CachePosJ=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),BJIndex);
        if (CachePosJ ==KernelCacheIndices.end())
        {
            CacheDiffJ=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
            d_KernelJ=d_Kernel_Cache+CacheDiffJ*m;
            mxCUDA_SAFE_CALL(cudaMemcpy(d_KernelInterRow, xT+BJIndex*n, n*sizeof(float),cudaMemcpyHostToDevice));
            RBFKernel(d_KernelJ,BJIndex,d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd,m,n,nbrCtas,threadsPerCta);
            *(KernelCacheIndices.begin()+CacheDiffJ)=BJIndex;
        }
        else
        {
            CacheDiffJ=CachePosJ-KernelCacheIndices.begin();
            d_KernelJ=d_Kernel_Cache+m*CacheDiffJ;

        }



        UpdateF<<<nbrCtas,threadsPerCta>>>(d_F,d_KernelI,d_KernelJ,d_y,deltaalphai,deltaalphaj,yi,yj,m);

        IncrementKernelCache(KernelCacheItersSinceUsed,RowsInKernelCache);

        *(KernelCacheItersSinceUsed.begin()+CacheDiffI)=0;
        *(KernelCacheItersSinceUsed.begin()+CacheDiffJ)=0;



        iter++;

        }

    cublasGetVector(m,sizeof(float),d_alpha,1,mexalpha,1);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    delete [] xT;
    cudaFreeHost(value_inter);
    cudaFreeHost(index_inter);

    mxCUDA_SAFE_CALL(cudaFree(d_x));
    mxCUDA_SAFE_CALL(cudaFree(d_y));
    mxCUDA_SAFE_CALL(cudaFree(d_alpha));
    mxCUDA_SAFE_CALL(cudaFree(d_KernelInterRow));
    mxCUDA_SAFE_CALL(cudaFree(d_Kernel_Cache));
    mxCUDA_SAFE_CALL(cudaFree(d_F));
    mxCUDA_SAFE_CALL(cudaFree(d_value_inter));
    mxCUDA_SAFE_CALL(cudaFree(d_index_inter));
    mxCUDA_SAFE_CALL(cudaFree(d_SelfDotProd));
    mxCUDA_SAFE_CALL(cudaFree(d_KernelDotProd));
    mxCUDA_SAFE_CALL(cudaThreadExit());

	*numSvs = *numPosSvs = 0;
	for(int k = 0; k < m; k++) {
		if(mexalpha[k]) {
			(*numSvs)++;
			mexalpha[k] *= y[k];
			if(y[k] > 0)
				(*numPosSvs)++;
		}
	}
*/
}

/*
 * This mixed-precision matrix-vector multiplication algorithm is based on
 * cublasSgemv NVIDIA's CUBLAS 1.1. In his tests, the author has found
 * catastrophic prediction errors resulting from using only single precision
 * floating point arithmetic for the multiplication of the predictive kernel
 * matrix by the SVM coefficients; however, all of the errors he found
 * disappeared when he switched to a mixed-precision approach where the scalar
 * dot-product accumulator is a double precision number.  Thus, the use of
 * full double precision arithmetic, which would involve significant
 * performance penalties, does not seem necessary. CUBLAS 1.1 source code is
 * available at: http://forums.nvidia.com/index.php?showtopic=59101, and
 * CUBLAS is available at http://www.nvidia.com/cuda
 */

#define LOG_THREAD_COUNT    (7)
#define THREAD_COUNT        (1 << LOG_THREAD_COUNT)
#define CTAS                (64)
#define IDXA(row,col)       (lda*(col)+(row))
#define IDXX(i)             (startx + ((i) * incx))
#define IDXY(i)             (starty + ((i) * incy))
#define TILEW_LOG           (5)
#define TILEW               (1 << TILEW_LOG)
#define TILEH_LOG           (5)
#define TILEH               (1 << TILEH_LOG)
#define X_ELEMS_PER_THREAD  (4)
#define IINC                (CTAS * THREAD_COUNT)
#define JINC                (THREAD_COUNT * X_ELEMS_PER_THREAD)
#define XINC                (THREAD_COUNT)


__shared__ float XX[TILEH];
__shared__ float AA[(TILEH+1)*TILEW];


__global__ void sgemvn_mixedprecis(const float *A, const float *x,float *y, int m, int n, int lda, int   incx,   int   incy)
{
    __shared__ float XX[JINC];
    int i, ii, j, jj, idx, incr, tid;
    double sdot;
    int startx;
    int starty;


    tid = threadIdx.x;
    startx = (incx >= 0) ? 0 : ((1 - n) * incx);
    starty = (incy >= 0) ? 0 : ((1 - m) * incy);

    for (i = 0; i < m; i += IINC) {

        ii = i + blockIdx.x * THREAD_COUNT;
        if (ii >= m) break;
        ii += tid;
        sdot = 0.0f;

        for (j = 0; j < n; j += JINC) {
            int jjLimit = min (j + JINC, n);
            incr = XINC * incx;
            jj = j + tid;
            __syncthreads ();
            idx = IDXX(jj);

            if (jj < (jjLimit - 3 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
                XX[tid+1*XINC] = x[idx + 1 * incr];
                XX[tid+2*XINC] = x[idx + 2 * incr];
                XX[tid+3*XINC] = x[idx + 3 * incr];
            }
            else if (jj < (jjLimit - 2 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
                XX[tid+1*XINC] = x[idx + 1 * incr];
                XX[tid+2*XINC] = x[idx + 2 * incr];
            }
            else if (jj < (jjLimit - 1 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
                XX[tid+1*XINC] = x[idx + 1 * incr];
            }
            else if (jj < (jjLimit - 0 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
            }

            __syncthreads ();

            if (ii < m) { /* if this row is active, accumulate dp */
                idx = IDXA(ii, j);
                incr = lda;
                jjLimit = jjLimit - j;
                jj = 0;
                while (jj < (jjLimit - 5)) {
                    sdot += A[idx + 0*incr] * XX[jj+ 0];
                    sdot += A[idx + 1*incr] * XX[jj+ 1];
                    sdot += A[idx + 2*incr] * XX[jj+ 2];
                    sdot += A[idx + 3*incr] * XX[jj+ 3];
                    sdot += A[idx + 4*incr] * XX[jj+ 4];
                    sdot += A[idx + 5*incr] * XX[jj+ 5];
                    jj   += 6;
                    idx  += 6 * incr;
                }
                while (jj < jjLimit) {
                    sdot += A[idx + 0*incr] * XX[jj+ 0];
                    jj   += 1;
                    idx  += 1 * incr;
                }
            }
        }
        if (ii < m) {
			idx = IDXY(ii);
			y[idx] = sdot;
        }
    }
}

// The memory access pattern and structure of this code is derived from
// Vasily Volkov's highly optimized matrix-matrix multiply CUDA code.
// His website is http://www.cs.berkeley.edu/~volkov/

__global__ void RBFKernelForPredict( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float kernelwidth )
{

    int inx = threadIdx.x;
	int iny = threadIdx.y;
	int ibx = blockIdx.x * 32;
	int iby = blockIdx.y * 32;

	A += ibx + inx + __mul24( iny, lda );
	B += iby + inx + __mul24( iny, ldb );
	C += ibx + inx + __mul24( iby + iny, ldc );

	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	for( int i = 0; i < k; i += 4 )
	{
		__shared__ float a[4][32];
		__shared__ float b[4][32];

		a[iny][inx] = A[i*lda];
		a[iny+2][inx] = A[(i+2)*lda];
		b[iny][inx] = B[i*ldb];
		b[iny+2][inx] = B[(i+2)*ldb];
		__syncthreads();

		for( int j = 0; j < 4; j++ )
		{
			float _a = a[j][inx];
			float *_b = &b[j][0] + iny;
            float _asquared=_a*_a;;


            //The (negative here) squared distance between datapoints is necessary for the calculation of the RBF Kernel.
            //This code uses the identity -(x-y)^2=2*x*y-x^2-y^2.

			c[0] += 2.f*_a*_b[0]-_asquared-_b[0]*_b[0];
			c[1] += 2.f*_a*_b[2]-_asquared-_b[2]*_b[2];
			c[2] += 2.f*_a*_b[4]-_asquared-_b[4]*_b[4];
			c[3] += 2.f*_a*_b[6]-_asquared-_b[6]*_b[6];
			c[4] += 2.f*_a*_b[8]-_asquared-_b[8]*_b[8];
			c[5] += 2.f*_a*_b[10]-_asquared-_b[10]*_b[10];
			c[6] += 2.f*_a*_b[12]-_asquared-_b[12]*_b[12];
			c[7] += 2.f*_a*_b[14]-_asquared-_b[14]*_b[14];
			c[8] += 2.f*_a*_b[16]-_asquared-_b[16]*_b[16];
			c[9] += 2.f*_a*_b[18]-_asquared-_b[18]*_b[18];
			c[10] += 2.f*_a*_b[20]-_asquared-_b[20]*_b[20];
			c[11] += 2.f*_a*_b[22]-_asquared-_b[22]*_b[22];
			c[12] += 2.f*_a*_b[24]-_asquared-_b[24]*_b[24];
			c[13] += 2.f*_a*_b[26]-_asquared-_b[26]*_b[26];
			c[14] += 2.f*_a*_b[28]-_asquared-_b[28]*_b[28];
			c[15] += 2.f*_a*_b[30]-_asquared-_b[30]*_b[30];
		}
		__syncthreads();
	}

	// Here the negative squared distances between datapoints,
	// calculated above, are multiplied by the kernel width
	// parameter and exponentiated.
	for( int i = 0; i < 16; i++, C += 2*ldc )
		C[0] = exp(kernelwidth*c[i]);
}

/*
 * Outputs:
 * 1. prediction is a single-precision vector of predictions of length m.
 * Inputs:
 * 1. test is a single-precision m by k matrix of test data.
 * 2. svs is the single-precision n by k matrix of support vectors output by
 *    SVMTrain SVRTrain.
 * 3. alphas is the single-precision vector of support vector coefficients
 *    output by cuSVMTrain of length n.
 * 4. beta is the single-precision scalar offset output by cuSVMTrain.
 * 5. kernel is the same scalar Gaussian kernel parameter value previously
 *    used in cuSVMTrain.
 * 6. regind is a scalar indicator variable that tells cuSVMPredict whether
 *    you are classifying or regressing. Set regind to 0 if the former and
 *    1 if the latter.
 */

void GPUPredictWrapper(int m, int n, int k, float kernelwidth,
	const float * Test, const float * Svs, float * alphas, float * prediction,
	float beta,float isregression)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    int paddedm=m+32-m%32;
    int paddedk=k+32-k%32;
    int paddedn=n+32-n%32;

    float* d_PaddedSvs;
    mxCUDA_SAFE_CALL( cudaMalloc( (void**) &d_PaddedSvs,
		paddedn*paddedk*sizeof(float)));

    mxCUDA_SAFE_CALL( cudaMemset(d_PaddedSvs, 0,
		paddedn * paddedk * sizeof(float)));
    mxCUDA_SAFE_CALL( cudaMemcpy(d_PaddedSvs, Svs,
		sizeof(float)*n*k,cudaMemcpyHostToDevice));

    float* d_PaddedSvsT;
    mxCUDA_SAFE_CALL( cudaMalloc( (void**) &d_PaddedSvsT,
		paddedn*paddedk*sizeof(float)));

    mxCUDA_SAFE_CALL(  cudaMemset(d_PaddedSvsT,0,
		paddedn*paddedk*sizeof(float)));

	int
		numBlocksX, numBlocksY;

	numBlocksX = (int) ceil((double) n / (double) TRANS_BLOCK_DIM);
	numBlocksY = (int) ceil((double) paddedk / (double) TRANS_BLOCK_DIM);

    dim3
		gridtranspose(numBlocksX, numBlocksY),
    	threadstranspose(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM);

    transpose<<<gridtranspose, threadstranspose>>>(d_PaddedSvsT, d_PaddedSvs,
		n, paddedk);

	numBlocksX = (int) ceil((double) paddedk / (double) TRANS_BLOCK_DIM);
	numBlocksY = (int) ceil((double) paddedn / (double) TRANS_BLOCK_DIM);

    dim3
		gridtranspose2(numBlocksX, numBlocksY);

    transpose<<<gridtranspose2, threadstranspose>>>(d_PaddedSvs, d_PaddedSvsT,
		paddedk,paddedn);
    mxCUDA_SAFE_CALL(  cudaFree(d_PaddedSvsT));

    double
		DoubleNecIterations = (double) paddedm / (double) CUBIC_ROOT_MAX_OPS;

    DoubleNecIterations *= (double) paddedn / (double) CUBIC_ROOT_MAX_OPS;
    DoubleNecIterations *= (double) paddedk / (double) CUBIC_ROOT_MAX_OPS;

    int
		NecIterations = (int) ceil(DoubleNecIterations),
    	RowsPerIter = (int) ceil((double) paddedm / (double) NecIterations)
			+ 32 - ((int) ceil((double) paddedm / (double) NecIterations))
			% 32;

    NecIterations = (int) ceil((double) paddedm / (double) RowsPerIter);

	dim3
		grid(RowsPerIter/32, paddedn/32, 1),
		threads2(32, 2, 1);

    float
		* d_TestInter, * d_QInter;

    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_TestInter,
		RowsPerIter*paddedk*sizeof(float)));

    mxCUDA_SAFE_CALL(cudaMemset(d_TestInter, 0,
		RowsPerIter*paddedk*sizeof(float)));

    mxCUDA_SAFE_CALL( cudaMalloc( (void**) &d_QInter,
		RowsPerIter*paddedn*sizeof(float)));

    float
		* d_alphas, * d_prediction;

    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_alphas, n*sizeof(float)));

    cublasSetVector(n,sizeof(float),alphas,1,d_alphas,1);

    mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_prediction,
		NecIterations*RowsPerIter*sizeof(float)));

    for(int j = 0; j < NecIterations; j++) {
        if (j + 1 == NecIterations) {
           cublasSetMatrix(m - j * RowsPerIter, k, sizeof(float),
		       Test + j * RowsPerIter, m, d_TestInter, RowsPerIter);
        } else {
            cublasSetMatrix(RowsPerIter, k, sizeof(float),
				Test + j * RowsPerIter, m, d_TestInter, RowsPerIter);
        }

        RBFKernelForPredict<<<grid, threads2>>>(d_TestInter, RowsPerIter,
			d_PaddedSvs, paddedn, d_QInter, RowsPerIter, paddedk,
			kernelwidth);

         sgemvn_mixedprecis<<<CTAS, THREAD_COUNT>>>(d_QInter,d_alphas,
		 	d_prediction+j*RowsPerIter,RowsPerIter,n,RowsPerIter,1,1);
    }
    cublasGetVector(m,sizeof(float),d_prediction,1,prediction,1);

    for(int j = 0; j < m; j++) {
        prediction[j]+=beta;
        if(isregression != 1.f)
			prediction[j] = (prediction[j] < 0)? -1.0:1.0;
    }

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    mxCUDA_SAFE_CALL(cudaFree(d_alphas));
    mxCUDA_SAFE_CALL(cudaFree(d_TestInter));
    mxCUDA_SAFE_CALL(cudaFree(d_QInter));
    mxCUDA_SAFE_CALL(cudaFree(d_PaddedSvs));
    mxCUDA_SAFE_CALL(cudaFree(d_prediction));

    mxCUDA_SAFE_CALL(cudaThreadExit());
}

double getAucEstimate(int n, double * classes, double * probs)
{
	double 
		* combined = Calloc(2 * n, double);

	memcpy(combined, classes, n * sizeof(double));
	memcpy(combined + n, probs, n * sizeof(double));

	quicksort(n, 2, 1, combined, 0, n - 1);

	double
		n0 = 0.0, n1 = 0.0, sum = 0.0;

	for(int i = 0; i < n; i++) {
		if(combined[i] == 0.f) {
			n0 = n0 + 1.0;
			sum = sum + (double) (i+1);
		} else {
			n1 = n1 + 1.0;
		}
	}
	Free(combined);
	return (sum - (n0 * (n0 + 1.0)) / 2.0) / (n0 * n1);
}

void R_SVRTrain(float * alpha, float * beta, float * y, float * x, float * C,
	float * kernelwidth, float * eps, int * m, int * n, float * StoppingCrit,
	int * numSvs)
{
	checkDoubleCapable("Your device doesn't support double precision arithmetic, so the SVM functionality is disabled. Sorry for any inconvenience.");

	SVRTrain(alpha, beta, y, x, *C, *kernelwidth, *eps, *m, *n,
		*StoppingCrit, numSvs);
}

void R_SVMTrain(float * alpha, float * beta, float * y, float * x, float * C,
	float * kernelwidth, int * m, int * n, float * StoppingCrit,
	int * numSvs, int * numPosSvs)
{
	checkDoubleCapable("Your device doesn't support double precision arithmetic, so the SVM functionality is disabled. Sorry for any inconvenience.");

	SVMTrain(alpha, beta, y, x, *C, *kernelwidth, *m, *n, *StoppingCrit,
		numSvs, numPosSvs);
}

void R_GPUPredictWrapper(int * m, int * n, int * k, float * kernelwidth,
	const float * Test, const float * Svs, float * alphas,
	float * prediction, float * beta, float * isregression)
{
	checkDoubleCapable("Your device doesn't support double precision arithmetic, so the SVM functionality is disabled. Sorry for any inconvenience.");

	GPUPredictWrapper(*m, *n, *k, *kernelwidth, Test, Svs, alphas, prediction,
		*beta, *isregression);
}

void R_produceSupportVectors(int * isRegression, int * m, int * n, int * numSVs,
	int * numPosSVs, const float * x, const float * y, const float * alphas,
	float * svCoefficients, float * supportVectors)
{
	checkDoubleCapable("Your device doesn't support double precision arithmetic, so the SVM functionality is disabled. Sorry for any inconvenience.");

	getSupportVectors(*isRegression, *m, *n, *numSVs, *numPosSVs, x, y,
		alphas, svCoefficients, supportVectors);
}

void RgetAucEstimate(int * n, double * classes, double * probs,
	double * outputAuc)
{
	*outputAuc = getAucEstimate(*n, classes, probs);
}
