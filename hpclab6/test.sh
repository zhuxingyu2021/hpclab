#!/usr/bin/env sh

EXEC_PATH=./GEMMcuda.x

if [ -d "./test.tmp" ]; then
  rm -rf ./test.tmp
fi

mkdir test.tmp

for num in $(seq 512 128 16384)
do
  file=test.tmp/output.txt
  touch $file
  $EXEC_PATH -M $num -N $num -K $num -m 1 -t 10 -e 1 >> $file
  echo >> $file
done


