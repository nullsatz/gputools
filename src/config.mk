R_HOME := $(shell R RHOME)
CUDA_HOME := $(CUDA_HOME)

R_INC := $(R_HOME)/include
R_LIB := $(R_HOME)/lib

CUDA_INC := $(CUDA_HOME)/include

ARCH := $(shell uname -m)
ifeq ($(ARCH), i386)
    CUDA_LIB := $(CUDA_HOME)/lib
else
    CUDA_LIB := $(CUDA_HOME)/lib64
endif

OS := $(shell uname -s)
ifeq ($(OS), Darwin)
    RPATHFLAG := -rpath,$(CUDA_LIB),$(R_LIB)
    ifeq ($(ARCH), x86_64)
        DEVICEOPTS := -m64
    endif
endif

CPICFLAGS := $(shell R CMD config CPICFLAGS)
