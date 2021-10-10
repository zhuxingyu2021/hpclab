CC_LIB_SRC=  \

CPP_LIB_SRC= \
	lib/walltime.cpp

CPP_MAIN_SRC1=GEMMpthreads.cpp
CPP_MAIN_SRC2=montecarlo.cpp
CPP_MAIN_SRC3=quadraticEquation.cpp
CPP_MAIN_SRC4=arraySum.cpp

INC=-Iinclude/

DEFS= \

CFLAGS=-O3 $(DEFS) $(INC)
CFLAGS_OBJ=
LDFLAGS= \
	-lpthread

LDFLAGS1=-lMYGEMM

LDLIBS=
LDLIBS1=-L${HOME}/lib/

LIBOBJS=$(CC_LIB_SRC:.c=.o) $(CPP_LIB_SRC:.cpp=.o)
ALLOBJS1=$(LIBOBJS) $(CPP_MAIN_SRC1:.cpp=.o)
ALLOBJS2=$(LIBOBJS) $(CPP_MAIN_SRC2:.cpp=.o)
ALLOBJS3=$(LIBOBJS) $(CPP_MAIN_SRC3:.cpp=.o)
ALLOBJS4=$(LIBOBJS) $(CPP_MAIN_SRC4:.cpp=.o)
OUTPUT1=$(CPP_MAIN_SRC1:.cpp=.x)
OUTPUT2=$(CPP_MAIN_SRC2:.cpp=.x)
OUTPUT3=$(CPP_MAIN_SRC3:.cpp=.x)
OUTPUT4=$(CPP_MAIN_SRC4:.cpp=.x)

all: $(OUTPUT1) $(OUTPUT2) $(OUTPUT3) $(OUTPUT4)
	rm -f $(ALLOBJS1) $(ALLOBJS2) $(ALLOBJS3) $(ALLOBJS4)

$(OUTPUT1): $(ALLOBJS1)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDFLAGS1) $(LDLIBS) $(LDLIBS1) -o $@ $^

$(OUTPUT2): $(ALLOBJS2)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDFLAGS1) $(LDLIBS) $(LDLIBS1) -o $@ $^

$(OUTPUT3): $(ALLOBJS3)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDFLAGS1) $(LDLIBS) $(LDLIBS1) -o $@ $^

$(OUTPUT4): $(ALLOBJS4)
	$(CXX) $(CFLAGS) -fuse-ld=lld $(LDFLAGS) $(LDFLAGS1) $(LDLIBS) $(LDLIBS1) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CFLAGS_OBJ) -c $< -o $@

%.so: $(LIBOBJS)
	$(CXX) $(CFLAGS) -fuse-ld=lld -fPIC -shared -o $@ $^

clean:
	rm -f $(ALLOBJS1) $(ALLOBJS2) $(ALLOBJS3) $(ALLOBJS4)
	rm -f $(OUTPUT1) $(OUTPUT2) $(OUTPUT3) $(OUTPUT4)
