#!/usr/bin/env sh

EXEC_PATH=./GEMMopenmp
COMMON_DEFINE_PATH=../include/common_define.h

if [ ! -d "./build-release" ]; then
  mkdir build-release
fi

cd build-release

if [ ! -e "./GEMMopenmp" ]; then
     cmake .. -DCMAKE_BUILD_TYPE=Release
fi

if [ ! -d "./output" ];
then
  mkdir output
else
  rm -rf ./output
fi

SCHEDULE_DYNAMIC="schedule(dynamic,1)"
SCHEDULE_STATIC="schedule(static,1)"
SCHEDULE_DEFAULT=" "

for thread in 1
  do
  mkdir output/nthread-$thread
  export OPENBLAS_NUM_THREADS=$thread
  echo "#ifndef _COMMON_DEFINE_H_" > $COMMON_DEFINE_PATH
  echo "#define _COMMON_DEFINE_H_" >> $COMMON_DEFINE_PATH
  echo "#define N_THREADS $thread" >> $COMMON_DEFINE_PATH
  echo "#define SCHEDULE " >> $COMMON_DEFINE_PATH
  echo "#endif" >> $COMMON_DEFINE_PATH
  make
  for num in 512 640 768 896 1024 1536 2048 3072 4096
  do
    file=output/nthread-$thread/$num-$num-$num.txt
    touch $file
    $EXEC_PATH -M $num -N $num -K $num -m 1 -t 10 >> $file
    echo >> $file
  done
done

for thread in 2 4 8
  do
  mkdir output/nthread-$thread
  export OPENBLAS_NUM_THREADS=$thread
  for schedulei in $(seq 1 3)
  do
    schedule=$SCHEDULE_DEFAULT
    if [ $schedulei -eq 1 ];
    then
      schedule=$SCHEDULE_STATIC
    else
      if [ $schedulei -eq 2 ];
      then
        schedule=$SCHEDULE_DYNAMIC
      fi
    fi
    echo "#ifndef _COMMON_DEFINE_H_" > $COMMON_DEFINE_PATH
    echo "#define _COMMON_DEFINE_H_" >> $COMMON_DEFINE_PATH
    echo "#define N_THREADS $thread" >> $COMMON_DEFINE_PATH
    echo "#define SCHEDULE $schedule" >> $COMMON_DEFINE_PATH
    echo "#endif" >> $COMMON_DEFINE_PATH
    make
    for num in 512 640 768 896 1024 1536 2048 3072 4096
    do
      file=output/nthread-$thread/$num-$num-$num.txt
      touch $file
      echo "SCHEDULE TYPE IS $schedule" >> $file
      $EXEC_PATH -M $num -N $num -K $num -m 1 -t 10 >> $file
      echo >> $file
    done
  done
done

