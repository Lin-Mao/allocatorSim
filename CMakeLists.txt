cmake_minimum_required(VERSION 3.16.3)
project(allcatorSim)

set(CMAKE_CXX_STANDARD 17)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src SRCS)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_executable(${PROJECT_NAME} ${SRCS})
