#!/usr/bin/env sh

EXEC_PATH=./GEMMwop

make

#if [ -d "./build-release" ]; then
#  rm -rf ./build-release
#fi

#mkdir build-release
cd build-release

if [ ! -e "./GEMMwop" ]; then
     cp ../GEMMwop ./
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_ENABLE_INSTRUCTIONS=AVX2
for num in $(seq 8 8 504)
do
  file=output.txt
  touch $file
  $EXEC_PATH -M $num -N $num -K $num -m 1 -t 2 -e 1 >> $file
  echo >> $file
done


