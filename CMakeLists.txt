cmake_minimum_required(VERSION 3.16.3)
project(libtorch_cuda_allocator_sim)
set(project_name ${PROJECT_NAME})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)

message(${SRCS})

add_library(${project_name} SHARED ${SRCS})

target_include_directories(${project_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)