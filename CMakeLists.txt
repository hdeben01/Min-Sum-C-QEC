cmake_minimum_required(VERSION 3.10.0)
project(minsum_c VERSION 0.1.0 LANGUAGES C)

include_directories(include)
#add_executable(minsum_c_csc min_sum_csc.c)
add_executable(minsum_c ./src_c/min_sum.c)
#configure_file(${CMAKE_SOURCE_DIR}/input/input.txt ${CMAKE_BINARY_DIR}/input.txt COPYONLY)
#configure_file(${CMAKE_SOURCE_DIR}/input/input2.txt ${CMAKE_BINARY_DIR}/input2.txt COPYONLY)
#configure_file(${CMAKE_SOURCE_DIR}/input/input3.txt ${CMAKE_BINARY_DIR}/input3.txt COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/input/input_sliding.txt ${CMAKE_BINARY_DIR}/input_sliding.txt COPYONLY)
#target_link_libraries(minsum_c_csc m)
include(CTest)
enable_testing()

