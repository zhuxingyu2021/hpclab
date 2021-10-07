cmake_minimum_required (VERSION 3.16)
project ("hpclab2")

set(CMAKE_CXX_STANDARD 14)

include_directories("include" "~/devtools/mpi/mpich3/include")

add_executable (GEMMmpi_p2p "GEMMmpi_p2p.cpp")
target_link_libraries(GEMMmpi_p2p "/home/zhuxingyu/libs/libMYGEMM.so")
target_link_libraries(GEMMmpi_p2p "~/devtools/mpi/mpich3/lib/libmpi.so")

add_executable (GEMMmpi_collection "GEMMmpi_collection.cpp")
target_link_libraries(GEMMmpi_collection  "/home/zhuxingyu/libs/libMYGEMM.so")
target_link_libraries(GEMMmpi_collection "~/devtools/mpi/mpich3/lib/libmpi.so")
