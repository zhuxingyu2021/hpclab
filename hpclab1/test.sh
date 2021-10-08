#!/usr/bin/env sh

EXEC_PATH=./GEMMwop

make

if [ ! -d "./build-release" ]; then
  mkdir build-release
fi

cd build-release

if [ ! -e "./GEMMwop" ]; then
     cp ../GEMMwop ./
fi

if [ ! -d "./output" ];
then
  mkdir output
else
  rm -rf ./output
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for num in 512 555 640 768 896 1024 1536 2048 3072 4096
do
  file=output/$num-$num-$num.txt
  touch $file
  $EXEC_PATH -M $num -N $num -K $num -m 1 -t 10 >> $file
  echo >> $file
done


