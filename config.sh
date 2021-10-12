export INTEL_ROOT=/opt/intel/oneapi
export INTEL_VERSION=2021.4.0
export MPI_ROOT=/home/zhuxingyu/devtools/mpi/mpich3

export INTEL_ROOT=/opt/intel/oneapi
export INTEL_VERSION=2021.4.0
export MPI_ROOT=/home/zhuxingyu/devtools/mpi/mpich3
export DEFS=-DINTEL_MKL

TOP_PATH=`pwd`
LIB_PATH=$TOP_PATH/lib

export LDLIBS=-L$LIB_PATH
export LDFLAGS=-Wl,-rpath=$LIB_PATH
