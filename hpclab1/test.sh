#!/usr/bin/env sh

EXEC_PATH=./GEMMwop

if [ ! -d "./build-release" ]; then
  mkdir build-release
fi

cd build-release

if [ ! -e "./GEMMwop" ]; then
     cmake .. -DCMAKE_BUILD_TYPE=Release
     make
fi

if [ ! -d "output" ];
then
  mkdir output
else
  rm -rf ./output
fi

export OPENBLAS_NUM_THREADS=1

for num in 512 555 640 896 1024
do
  for i in $(seq 1 10)
  do
    touch output/$num-$num-$num.txt
    $EXEC_PATH -M $num -N $num -K $num >> output/$num-$num-$num.txt
    echo >> output/$num-$num-$num.txt
  done
done

for num in 1536 2048
do
  for i in $(seq 1 5)
  do
    touch output/$num-$num-$num.txt
    $EXEC_PATH -M $num -N $num -K $num >> output/$num-$num-$num.txt
    echo >> output/$num-$num-$num.txt
  done
done

for num in 3072 4096
do
  for i in $(seq 1 2)
  do
    touch output/$num-$num-$num.txt
    $EXEC_PATH -M $num -N $num -K $num >> output/$num-$num-$num.txt
    echo >> output/$num-$num-$num.txt
  done
done

