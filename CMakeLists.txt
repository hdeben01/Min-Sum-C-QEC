cmake_minimum_required(VERSION 3.10.0)
project(minsum_c VERSION 0.1.0 LANGUAGES C)

include_directories(include)
add_executable(minsum_c main.c)
configure_file(${CMAKE_SOURCE_DIR}/input.txt ${CMAKE_BINARY_DIR}/input.txt COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/input2.txt ${CMAKE_BINARY_DIR}/input2.txt COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/input3.txt ${CMAKE_BINARY_DIR}/input3.txt COPYONLY)
include(CTest)
enable_testing()

