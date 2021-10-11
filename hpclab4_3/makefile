CC=icx
CXX=icpx

CC_LIB_SRC=  \

CPP_LIB_SRC= \
	lib/GEMM_fast.cpp\
	lib/GEMM_naive.cpp\
	lib/GEMM_utils.cpp\
	lib/walltime.cpp

CC_MAIN_SRC= \

CPP_MAIN_SRC= \
	GEMMparallelfor.cpp

INC=-Iinclude/\
	-I${INTEL_ROOT}/mkl/${INTEL_VERSION}/include

MKL_LIB=${INTEL_ROOT}/mkl/${INTEL_VERSION}/lib/intel64

DEFS=-DINTEL_MKL

CFLAGS=-O3 -march=core-avx2 $(DEFS) $(INC)
CFLAGS_OBJ=-fPIC
LDFLAGS+=-fopenmp -qmkl=sequential -lparallelfor -Wl,-rpath=$(MKL_LIB)
LDLIBS+=-L$(MKL_LIB)\
	-L.

LIBOBJS=$(CC_LIB_SRC:.c=.o) $(CPP_LIB_SRC:.cpp=.o)
ALLOBJS=$(LIBOBJS) $(CC_MAIN_SRC:.c=.o) $(CPP_MAIN_SRC:.cpp=.o)
OUTPUT=$(CPP_MAIN_SRC:.cpp=.x)
OUTPUT_LIBOBJS=lib/parallelfor.o
OUTPUT_LIB=libparallelfor.so

all: $(OUTPUT) $(OUTPUT_LIB)
	rm $(ALLOBJS)

$(OUTPUT): $(ALLOBJS) $(OUTPUT_LIB)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDLIBS) -o $@ $(ALLOBJS)

%.o: %.c
	$(CC) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.so: $(OUTPUT_LIBOBJS)
	$(CXX) $(CFLAGS) -shared -fPIC -o $@ $^

clean:
	rm -f $(ALLOBJS)
	rm -f $(OUTPUT)
	rm -f $(OUTPUT_LIB)
