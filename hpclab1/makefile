CC=icx
CXX=icpx
LD=ld.lld

CC_LIB_SRC=  \

CPP_LIB_SRC= \
	lib/GEMM_fast.cpp\
	lib/GEMM_naive.cpp\
	lib/GEMM_utils.cpp

CC_MAIN_SRC= \

CPP_MAIN_SRC= \
	GEMMwop.cpp

INTEL_ROOT=/opt/intel/oneapi

INC=-Iinclude/\
	-I${INTEL_ROOT}/mkl/2021.4.0/include

MKL_LIB=${INTEL_ROOT}/mkl/2021.4.0/lib/intel64

DEFS=-DINTEL_MKL

CFLAGS=-O3 -march=core-avx2 $(DEFS) $(INC)
LDFLAGS=-lmkl_intel_thread -lmkl_core -lmkl_intel_ilp64 -fopenmp
LDLIBS=-L$(MKL_LIB)

LIBOBJS=$(CC_LIB_SRC:.c=.o) $(CPP_LIB_SRC:.cpp=.o)
ALLOBJS=$(LIBOBJS) $(CC_MAIN_SRC:.c=.o) $(CPP_MAIN_SRC:.cpp=.o)
OUTPUT=GEMMwop libMYGEMM.so

all: $(OUTPUT)

$(OUTPUT): $(ALLOBJS)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDLIBS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

%.so: $(LIBOBJS)
	$(CXX) $(CFLAGS) -fuse-ld=lld -fPIC -shared -o $@ $^

clean:
	rm $(ALLOBJS)
	rm $(OUTPUT)