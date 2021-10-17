.PHONY: clean

export INTEL_ROOT=/opt/intel/oneapi
export INTEL_VERSION=2021.4.0
export MPI_ROOT=$(HOME)/mpich
export CUDA_PATH=/usr/local/cuda
export DEFS=-DINTEL_MKL

TOP_PATH=$(shell pwd)
LIB_PATH=$(TOP_PATH)/lib

export LDLIBS=-L$(LIB_PATH)
export LDFLAGS=-Wl,-rpath=$(LIB_PATH)

SUBDIR1=hpclab1
SUBDIR2=hpclab2
SUBDIR3=hpclab3
SUBDIR4=hpclab4_12
SUBDIR5=hpclab4_3
SUBDIR6=hpclab5
SUBDIR7=hpclab6
SUBDIR8=hpclab6_omp

TARGET1_SO=$(LIB_PATH)/libMYGEMM.so
TARGET5_SO=$(LIB_PATH)/libparallelfor.so

SOURCE1_CPP=$(SUBDIR1)/$(shell ls $(SUBDIR1)|grep '.cpp$$|awk 'NR==1'')
SOURCE2_CPP=$(SUBDIR2)/$(shell ls $(SUBDIR2)|grep '.cpp$$|awk 'NR==1'')
SOURCE3_CPP=$(SUBDIR3)/$(shell ls $(SUBDIR3)|grep '.cpp$$|awk 'NR==1'')
SOURCE4_CPP=$(SUBDIR4)/$(shell ls $(SUBDIR4)|grep '.cpp$$|awk 'NR==1'')
SOURCE5_CPP=$(SUBDIR5)/$(shell ls $(SUBDIR5)|grep '.cpp$$|awk 'NR==1'')
SOURCE6_CPP=$(SUBDIR6)/$(shell ls $(SUBDIR6)|grep '.cpp$$|awk 'NR==1'')
SOURCE7_CPP=$(SUBDIR7)/$(shell ls $(SUBDIR7)|grep '.cpp$$|awk 'NR==1'')
SOURCE8_CPP=$(SUBDIR8)/$(shell ls $(SUBDIR8)|grep '.cpp$$|awk 'NR==1'')

TARGET1=$(SOURCE1_CPP:.cpp=.x)
TARGET2=$(SOURCE2_CPP:.cpp=.x)
TARGET3=$(SOURCE3_CPP:.cpp=.x)
TARGET4=$(SOURCE4_CPP:.cpp=.x)
TARGET5=$(SOURCE5_CPP:.cpp=.x)
TARGET6=$(SOURCE6_CPP:.cpp=.x)
TARGET7=$(SOURCE7_CPP:.cpp=.x)
TARGET8=$(SOURCE8_CPP:.cpp=.x)

all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) $(TARGET7) $(TARGET8)

$(TARGET1):$(LIB_PATH)
	cd $(SUBDIR1) && $(MAKE) && cd ..
	cp $(SUBDIR1)/*.so $(LIB_PATH)
	rm -f $(SUBDIR1)/*.so

$(TARGET2):$(LIB_PATH) $(TARGET1_SO)
	cd $(SUBDIR2) && $(MAKE) && cd ..

$(TARGET3):$(LIB_PATH) $(TARGET1_SO)
	cd $(SUBDIR3) && $(MAKE) && cd ..

$(TARGET4):$(LIB_PATH)
	cd $(SUBDIR4) && $(MAKE) && cd ..

$(TARGET5):$(LIB_PATH)
	cd $(SUBDIR5) && $(MAKE) && cd ..
	cp $(SUBDIR5)/*.so $(LIB_PATH)
	rm -f $(SUBDIR5)/*.so

$(TARGET6):$(LIB_PATH) $(TARGET5_SO)
	cd $(SUBDIR6) && $(MAKE) && cd ..

$(TARGET7):$(LIB_PATH)
	cd $(SUBDIR7) && $(MAKE) && cd ..

$(TARGET8):$(LIB_PATH)
	cd $(SUBDIR8) && $(MAKE) && cd ..

$(TARGET1_SO):$(TARGET1)

$(TARGET5_SO):$(TARGET5)

$(LIB_PATH):
	mkdir $(LIB_PATH)

clean:
	cd $(SUBDIR1) && $(MAKE) clean && cd ..
	cd $(SUBDIR2) && $(MAKE) clean && cd ..
	cd $(SUBDIR3) && $(MAKE) clean && cd ..
	cd $(SUBDIR4) && $(MAKE) clean && cd ..
	cd $(SUBDIR5) && $(MAKE) clean && cd ..
	cd $(SUBDIR6) && $(MAKE) clean && cd ..
	cd $(SUBDIR7) && $(MAKE) clean && cd ..
	cd $(SUBDIR8) && $(MAKE) clean && cd ..
	rm -rf $(LIB_PATH)
