CC_LIB_SRC=  \

CPP_LIB_SRC= \

CPP_MAIN_SRC1=GEMMmpi_collection.cpp
CPP_MAIN_SRC2=GEMMmpi_p2p.cpp

INC=-Iinclude/\
	-I${MPI_ROOT}/include

DEFS= \

MPI_LIB=${MPI_ROOT}/lib/

CFLAGS=-O3 $(DEFS) $(INC)
CFLAGS_OBJ=

LDFLAGS+= \
	-lMYGEMM \
	-lmpi \
	-Wl,-rpath=-L$(MPI_LIB)

LDLIBS+=-L$(MPI_LIB)

LIBOBJS=$(CC_LIB_SRC:.c=.o) $(CPP_LIB_SRC:.cpp=.o)
ALLOBJS1=$(LIBOBJS) $(CPP_MAIN_SRC1:.cpp=.o)
ALLOBJS2=$(LIBOBJS) $(CPP_MAIN_SRC2:.cpp=.o)
OUTPUT1=$(CPP_MAIN_SRC1:.cpp=.x)
OUTPUT2=$(CPP_MAIN_SRC2:.cpp=.x)

all: $(OUTPUT1) $(OUTPUT2)
	rm $(ALLOBJS1) $(ALLOBJS2)

$(OUTPUT1): $(ALLOBJS1)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDLIBS) -o $@ $^

$(OUTPUT2): $(ALLOBJS2)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDLIBS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.so: $(LIBOBJS)
	$(CXX) $(CFLAGS) -fuse-ld=lld -fPIC -shared -o $@ $^

clean:
	rm -f $(ALLOBJS1) $(ALLOBJS2)
	rm -f $(OUTPUT1)
	rm -f $(OUTPUT2)
