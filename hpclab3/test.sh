#!/usr/bin/env sh

EXEC_PATH_GEMM=./GEMMpthreads

if [ ! -d "./build-release" ]; then
  mkdir build-release
fi

cd build-release

if [ ! -e "./GEMMpthreads" ]; then
     cmake .. -DCMAKE_BUILD_TYPE=Release
     make
fi

if [ ! -d "output" ];
then
  mkdir output
else
  rm -rf ./output
fi

for num in 512 768 1024 1536 2048 3072 4096 8192
do
  file=output/$num-$num-$num.txt
  touch file
  for proc in 1 2 4 8
  do
    echo "The number of threads is $proc" >> $file
    for i in $(seq 1 10)
    do
      $EXEC_PATH_GEMM -M $num -N $num -K $num -n $proc -s 1 >> $file
      echo >> $file
    done
  done
done

