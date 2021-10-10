cmake_minimum_required(VERSION 3.16)
project(hpclab1)

set(CMAKE_CXX_STANDARD 14)

add_compile_options("-march=core-avx2")

add_library(MYGEMM SHARED "lib/GEMM_fast.cpp" "lib/GEMM_naive.cpp" "lib/GEMM_utils.cpp")

include_directories("include")

add_executable (GEMMwop "GEMMwop.cpp")
target_link_libraries(GEMMwop MYGEMM)


set(OPEN_BLAS_PATH "$ENV{HOME}/devtools/openblas")
target_include_directories(GEMMwop PRIVATE ${OPEN_BLAS_PATH}/include)
target_link_libraries(GEMMwop ${OPEN_BLAS_PATH}/lib/libopenblas.so)


