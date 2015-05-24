OBJS = rinterface.o mi.o sort.o granger.o qrdecomp.o correlation.o hcluster.o distance.o matmult.o lsfit.o kendall.o cuseful.o

NVCC = @NVCC@
CUDA_HOME = @CUDA_HOME@
CUDA_INCLUDE = $(CUDA_HOME)/include
CUDA_LIB = $(CUDA_HOME)/lib

R_HOME = @R_HOME@
R_INCLUDE = @R_INCLUDE@
R_LIB = @R_LIB@

#compiler/preprocessor options
INCLUDES = -I. -I"$(CUDA_INCLUDE)" -I"$(R_INCLUDE)"
LIBS =  -L"$(R_LIB)" -lR -L"$(CUDA_LIB)" -lcublas

TARGETS := gputools.so

all: $(TARGETS) 

$(TARGETS): $(OBJS)
	$(NVCC) -shared $(LIBS) $(OBJS) -o $@

$(OBJS): %.o: %.$(EXT)
	$(NVCC) -c $(INCLUDES) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
