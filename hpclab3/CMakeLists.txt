cmake_minimum_required(VERSION 3.16)
project(hpclab3)

set(CMAKE_CXX_STANDARD 14)

add_compile_options("-lpthread")

include_directories("include")
link_libraries("/usr/lib/x86_64-linux-gnu/libpthread.so")

add_executable(GEMMpthreads GEMMpthreads.cpp lib/walltime.cpp)
target_link_libraries(GEMMpthreads "/home/zhuxingyu/libs/libMYGEMM.so")

add_executable(arraySum arraySum.cpp lib/walltime.cpp)
add_executable(quadraticEquation quadraticEquation.cpp)
add_executable(montecarlo montecarlo.cpp lib/walltime.cpp)
