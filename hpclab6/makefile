NVCC=nvcc

CC_LIB_SRC=  \

CPP_LIB_SRC= \
	lib/cugemm_utils.cpp

CU_LIB_SRC= \
	lib/cugemm_fast.cu

CC_MAIN_SRC= \

CPP_MAIN_SRC= \
	GEMMcuda.cpp

CPP_MAIN_SRC_MPI=\
	GEMMcuda_mpi.cpp

INC=-Iinclude\
    -I$(CUDA_PATH)/include\
    -I${MPI_ROOT}/include


DEFS+=

CFLAGS=-O3 $(DEFS) $(INC)
NVCC_CFLAGS=$(CFLAGS) -arch=compute_70 -code=sm_70

LDFLAGS+=-lcuda -lcudart -lcublas
LDFLAGSMPI=$(LDFLAGS)\
	-lmpi \
	-Wl,-rpath=$(MPI_LIB)

LDLIBS+=-L$(CUDA_PATH)/lib64

MPI_LIB=${MPI_ROOT}/lib/
LDLIBSMPI=$(LDLIBS) -L$(MPI_LIB)

LIBOBJS=$(CC_LIB_SRC:.c=.o) $(CPP_LIB_SRC:.cpp=.o) $(CU_LIB_SRC:.cu=.o)
ALLOBJS=$(LIBOBJS) $(CC_MAIN_SRC:.c=.o) $(CPP_MAIN_SRC:.cpp=.o)
OUTPUT=$(CPP_MAIN_SRC:.cpp=.x)
OUTPUTMPI=$(CPP_MAIN_SRC_MPI:.cpp=.x)
ALLOBJSMPI=$(LIBOBJS) $(CPP_MAIN_SRC_MPI:.cpp=.o)

all: $(OUTPUT) $(OUTPUTMPI)
	rm $(ALLOBJS)
	rm -f *.tmp

$(OUTPUT): $(ALLOBJS)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDLIBS) -o $@ $^

$(OUTPUTMPI): $(ALLOBJSMPI)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGSMPI) $(LDLIBSMPI) -o $@ $^

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
