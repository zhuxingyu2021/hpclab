cmake_minimum_required(VERSION 3.16)
project(hpclab1)

set(CMAKE_CXX_STANDARD 14)

add_compile_options("-march=core-avx2")
add_compile_options("-lpthread")

include_directories("include")
link_libraries("/usr/lib/x86_64-linux-gnu/libpthread.so")

add_library(parallelfor SHARED "lib/parallelfor.cpp")

add_executable (GEMMparallelfor "GEMMparallelfor.cpp" "lib/walltime.cpp" "lib/GEMM_fast.cpp" "lib/GEMM_naive.cpp" "lib/GEMM_utils.cpp")
target_link_libraries(GEMMparallelfor parallelfor)

set(OPEN_BLAS_PATH "$ENV{HOME}/devtools/openblas")
target_include_directories(GEMMparallelfor PRIVATE ${OPEN_BLAS_PATH}/include)
target_link_libraries(GEMMparallelfor ${OPEN_BLAS_PATH}/lib/libopenblas.so)
