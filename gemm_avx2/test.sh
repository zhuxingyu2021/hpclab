#!/usr/bin/env sh

EXEC_PATH=./GEMMwop.x

if [ -d "./test.tmp" ]; then
  rm -rf ./test.tmp
fi

mkdir test.tmp

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_ENABLE_INSTRUCTIONS=AVX2
for num in $(seq 8 8 2048)
do
  file=test.tmp/output.txt
  touch $file
  $EXEC_PATH -M $num -N $num -K $num -m 1 -t 2 -e 1 >> $file
  echo >> $file
done


