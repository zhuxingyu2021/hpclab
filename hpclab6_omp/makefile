NVCC=nvcc
CXX=icpx
CC=icpc

CC_LIB_SRC=  \

CPP_LIB_SRC= \
	lib/cugemm_utils.cpp\
	lib/walltime.cpp \
	lib/cugemm_fast_multithread.cpp \
	lib/cugemm_fast_singlethread.cpp \
	lib/cugemm_fast_multistream.cpp

CU_LIB_SRC= \
	lib/cugemm_fast_kernel.cu

CC_MAIN_SRC= \

CPP_MAIN_SRC= \
	GEMMcuda_omp.cpp

INC=-Iinclude\
    -I$(CUDA_PATH)/include

DEFS+=

CFLAGS=-g $(DEFS) $(INC)
NVCC_CFLAGS=$(CFLAGS) -arch=compute_70 -code=sm_70 --default-stream per-thread

LDFLAGS+=-lcuda -lcudart -lcublas -fopenmp

LDLIBS+=-L$(CUDA_PATH)/lib64

LIBOBJS=$(CC_LIB_SRC:.c=.o) $(CPP_LIB_SRC:.cpp=.o) $(CU_LIB_SRC:.cu=.o)
ALLOBJS=$(LIBOBJS) $(CC_MAIN_SRC:.c=.o) $(CPP_MAIN_SRC:.cpp=.o)
OUTPUT=$(CPP_MAIN_SRC:.cpp=.x)

all: $(OUTPUT)
	rm $(ALLOBJS)
	rm -f *.tmp

$(OUTPUT): $(ALLOBJS)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDLIBS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) -fopenmp -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCC_CFLAGS) -c $< -o $@
    
%.ptx:%.cu
	$(NVCC) $(NVCC_CFLAGS) -c $< -o $@ -ptx

clean:
	rm -f $(ALLOBJS)
	rm -f $(OUTPUT)
	rm -f *.tmp