include config.mk

EXT := cu

OBJS := rinterface.o mi.o sort.o granger.o qrdecomp.o correlation.o hcluster.o distance.o matmult.o lsfit.o kendall.o cuseful.o

#compiler/preprocessor options
INCS := -I. -I"$(CUDA_INC)" -I"$(R_INC)"
PARAMS := $(DEVICEOPTS) -Xcompiler $(CPICFLAGS)

#linker options
LD_PARAMS := $(DEVICEOPTS) -Xlinker '$(RPATH) $(R_FRAMEWORK)'
LIBS :=  -L"$(R_LIB)" -L"$(CUDA_LIB)" -lcublas $(shell $(R_HOME)/bin/R CMD config BLAS_LIBS)

TARGETS := gputools.so

# Compute capabilities 1.0 and 1.3 are obsolete
# Add compute capabilities 3.5, 3.7, 5.0, 5.2
NVCC := $(CUDA_HOME)/bin/nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52

all: $(TARGETS)

$(TARGETS): $(OBJS)
	$(NVCC) -shared $(LD_PARAMS) $(LIBS) $(OBJS) -o $@

$(OBJS): %.o: %.$(EXT)
	$(NVCC) -c $(INCS) $(PARAMS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
