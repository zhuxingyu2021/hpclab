cmake_minimum_required(VERSION 3.16)
project(hpclab5)

set(CMAKE_CXX_STANDARD 14)

add_compile_options("-march=core-avx2")
add_compile_options("-lpthread")

link_libraries("/usr/lib/x86_64-linux-gnu/libpthread.so")

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("Found OpenMP")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
endif()

include_directories("include" "~/devtools/mpi/mpich3/include")

add_executable(heated_plate_openmp heated_plate_openmp.c)
add_executable(heated_plate_pf heated_plate_pf.cpp lib/walltime.cpp lib/parallelfor.cpp)
add_executable(heated_plate_pf_cyclic heated_plate_pf_cyclic.cpp lib/walltime.cpp lib/parallelfor.cpp)
add_executable(heated_plate_pf_wo_thread_pool heated_plate_pf_wo_thread_pool.cpp lib/walltime.cpp lib/parallelfor.cpp)
add_executable(heated_plate_mpi heated_plate_mpi.cpp)
target_link_libraries(heated_plate_mpi ~/devtools/mpi/mpich3/lib/libmpi.so)
#target_link_libraries(heated_plate_pf /home/zhuxingyu/libs/libparallelfor.so)
#target_link_libraries(heated_plate_pf_wo_thread_pool /home/zhuxingyu/libs/libparallelfor.so)
