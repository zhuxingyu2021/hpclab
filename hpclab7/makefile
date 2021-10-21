NVCC=nvcc

CC_LIB_SRC=  \

CPP_LIB_SRC= \
	lib/cuconv_utils.cpp

CU_LIB_SRC= \
	lib/cuconv_naive.cu

CC_MAIN_SRC= \

CPP_MAIN_SRC= \
	cudaconv.cpp

INC=-Iinclude\
    -I$(CUDA_PATH)/include


DEFS+=

CFLAGS=-O3 $(DEFS) $(INC)
NVCC_CFLAGS=$(CFLAGS) -arch=compute_70 -code=sm_70

LDFLAGS+=-lcuda -lcudart -lcublas

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
	$(CXX) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCC_CFLAGS) -c $< -o $@
    
%.ptx:%.cu
	$(NVCC) $(NVCC_CFLAGS) -c $< -o $@ -ptx

clean:
	rm -f $(ALLOBJS)
	rm -f $(OUTPUT)
	rm -f *.tmp