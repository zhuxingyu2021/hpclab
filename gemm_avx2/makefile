CC=icx
CXX=icpx

CC_LIB_SRC=  \

CPP_LIB_SRC= \
	lib/GEMM_fast.cpp\
	lib/GEMM_naive.cpp\
	lib/GEMM_utils.cpp\


CC_MAIN_SRC= \

CPP_MAIN_SRC= \
	GEMMwop.cpp

INC=-Iinclude/\
	-I${INTEL_ROOT}/mkl/${INTEL_VERSION}/include

MKL_LIB=${INTEL_ROOT}/mkl/${INTEL_VERSION}/lib/intel64

DEFS+=

CFLAGS=-O3 -march=core-avx2 $(DEFS) $(INC)
CFLAGS_OBJ=-fPIC
LDFLAGS+=-fopenmp -qmkl=sequential -Wl,-rpath=$(MKL_LIB) \
	-Wl,-rpath=${INTEL_ROOT}/compiler/${INTEL_VERSION}/linux/compiler/lib/intel64_lin/
LDLIBS+=-L$(MKL_LIB)

LIBOBJS=$(CC_LIB_SRC:.c=.o) $(CPP_LIB_SRC:.cpp=.o)
ALLOBJS=$(LIBOBJS) $(CC_MAIN_SRC:.c=.o) $(CPP_MAIN_SRC:.cpp=.o)
OUTPUT=$(CPP_MAIN_SRC:.cpp=.x)
OUTPUT_LIB=libMYGEMM.so

all: $(OUTPUT) $(OUTPUT_LIB)
	rm $(ALLOBJS)
	rm -f *.tmp

$(OUTPUT): $(ALLOBJS)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDLIBS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.so: $(LIBOBJS)
	$(CXX) $(CFLAGS) -shared -fPIC -o $@ $^

clean:
	rm -f $(ALLOBJS)
	rm -f $(OUTPUT)
	rm -f $(OUTPUT_LIB)
	rm -f *.tmp
