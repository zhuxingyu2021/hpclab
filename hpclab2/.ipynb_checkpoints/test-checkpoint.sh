#!/usr/bin/env sh

EXEC_PATH_COLLECTION=./GEMMmpi_collection
EXEC_PATH_P2P=./GEMMmpi_p2p

if [ ! -d "./build-release" ]; then
  mkdir build-release
fi

cd build-release

if [ ! -e "./GEMMmpi_collection" ]; then
     cmake .. -DCMAKE_BUILD_TYPE=Release
     make
fi

if [ ! -d "output" ];
then
  mkdir output
else
  rm -rf ./output
fi

for num in 512 768 1024 1536 2048
do
  file=output/$num-$num-$num-p2p.txt
  touch $file
  for proc in 1 2 4 8
  do
    echo "The number of processes is $proc" >> $file
    for i in $(seq 1 10)
    do
      mpiexec -n $proc $EXEC_PATH_P2P -M $num -N $num -K $num -s 1 >> $file
      echo >> $file
    done
  done
done

for num in 3072 4096 8192
do
  file=output/$num-$num-$num-p2p.txt
  touch $file
  for proc in 1 2 4 8
  do
    echo "The number of processes is $proc" >> $file
    for i in $(seq 1 5)
    do
      mpiexec -n $proc $EXEC_PATH_P2P -M $num -N $num -K $num -s 1 >> $file
      echo >> $file
    done
  done
done

for num in 512 768 1024 1536 2048
do
  file=output/$num-$num-$num-collection.txt
  touch $file
  for proc in 1 2 4 8
  do
    echo "The number of processes is $proc" >> $file
    for i in $(seq 1 10)
    do
      mpiexec -n $proc $EXEC_PATH_COLLECTION -M $num -N $num -K $num -s 1 >> $file
      echo >> $file
    done
  done
done

for num in 3072 4096 8192
do
  file=output/$num-$num-$num-collection.txt
  touch $file
  for proc in 1 2 4 8
  do
    echo "The number of processes is $proc" >> $file
    for i in $(seq 1 5)
    do
      mpiexec -n $proc $EXEC_PATH_COLLECTION -M $num -N $num -K $num -s 1 >> $file
      echo >> $file
    done
  done
done
