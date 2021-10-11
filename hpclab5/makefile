CC_LIB_SRC=  \

CPP_LIB_SRC= \
	lib/walltime.cpp

CPP_MAIN_SRC1=heated_plate_mpi.cpp
CPP_MAIN_SRC2=heated_plate_openmp.cpp
CPP_MAIN_SRC3=heated_plate_pf.cpp
CPP_MAIN_SRC4=heated_plate_pf_cyclic.cpp
CPP_MAIN_SRC5=heated_plate_pf_wo_thread_pool.cpp

INC=-Iinclude/\
	-I${MPI_ROOT}/include

DEFS+= \

MPI_LIB=${MPI_ROOT}/lib/

CFLAGS=-O3 $(DEFS) $(INC)
CFLAGS_OBJ=
LDFLAGS+=
LDFLAGS1=$(LDFLAGS) -lmpi -Wl,-rpath=$(MPI_LIB)
LDFLAGS2=$(LDFLAGS) -fopenmp
LDFLAGS3=$(LDFLAGS) -lparallelfor -lpthread
LDFLAGS4=$(LDFLAGS) -lparallelfor -lpthread
LDFLAGS5=$(LDFLAGS) -lparallelfor -lpthread

LDLIBS+=
LDLIBS1=$(LDLIBS) -L$(MPI_LIB)
LDLIBS2=$(LDLIBS)
LDLIBS3=$(LDLIBS)
LDLIBS4=$(LDLIBS)
LDLIBS5=$(LDLIBS)


LIBOBJS=$(CC_LIB_SRC:.c=.o) $(CPP_LIB_SRC:.cpp=.o)
ALLOBJS1=$(LIBOBJS) $(CPP_MAIN_SRC1:.cpp=.o)
ALLOBJS2=$(LIBOBJS) $(CPP_MAIN_SRC2:.cpp=.o)
ALLOBJS3=$(LIBOBJS) $(CPP_MAIN_SRC3:.cpp=.o)
ALLOBJS4=$(LIBOBJS) $(CPP_MAIN_SRC4:.cpp=.o)
ALLOBJS5=$(LIBOBJS) $(CPP_MAIN_SRC5:.cpp=.o)
OUTPUT1=$(CPP_MAIN_SRC1:.cpp=.x)
OUTPUT2=$(CPP_MAIN_SRC2:.cpp=.x)
OUTPUT3=$(CPP_MAIN_SRC3:.cpp=.x)
OUTPUT4=$(CPP_MAIN_SRC4:.cpp=.x)
OUTPUT5=$(CPP_MAIN_SRC5:.cpp=.x)

all: $(OUTPUT1) $(OUTPUT2) $(OUTPUT3) $(OUTPUT4) $(OUTPUT5)
	rm -f $(ALLOBJS1) $(ALLOBJS2) $(ALLOBJS3) $(ALLOBJS4) $(ALLOBJS5)
	rm -f *.tmp

$(OUTPUT1): $(ALLOBJS1)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS1) $(LDLIBS1) -o $@ $^

$(OUTPUT2): $(ALLOBJS2)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS2) $(LDLIBS2) -o $@ $^

$(OUTPUT3): $(ALLOBJS3)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS3) $(LDLIBS3) -o $@ $^

$(OUTPUT4): $(ALLOBJS4)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS4) $(LDLIBS4) -o $@ $^

$(OUTPUT5): $(ALLOBJS5)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS5) $(LDLIBS5) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.so: $(LIBOBJS)
	$(CXX) $(CFLAGS) -fuse-ld=lld -fPIC -shared -o $@ $^

clean:
	rm -f $(ALLOBJS1) $(ALLOBJS2) $(ALLOBJS3) $(ALLOBJS4) $(ALLOBJS5)
	rm -f $(OUTPUT1) $(OUTPUT2) $(OUTPUT3) $(OUTPUT4) $(OUTPUT5)
	rm -f *.tmp