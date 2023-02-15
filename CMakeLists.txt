cmake_minimum_required(VERSION 3.16)

set(C10_CUDA_SIMULATOR_SRCS
    src/allocator_config.cpp
    src/allocator_simulator.cpp
    src/allocator_manager.cpp
    src/allocator_profiler.cpp
    src/allocator_utils.cpp
    src/python_states.cpp
)

if(NOT TORCH_INSTALL_LIB_DIR)
    set(TORCH_INSTALL_LIB_DIR lib)
endif()
message(STATUS "TORCH_INSTALL_LIB_DIR: ${TORCH_INSTALL_LIB_DIR}")

find_package(Python COMPONENTS Interpreter Development)
if (NOT ${Python_FOUND})
    message(SEND_ERROR "PYTHON not found")
endif()

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(PYTHON_LINKER_FLAGS "-L${Python_LIBRARY_DIRS} -Wl,-rpath=${Python_LIBRARY_DIRS} -lpython3.8")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${PYTHON_LINKER_FLAGS}")
set(CMAKE_CXX_STANDARD 17)

set(PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
add_subdirectory(${PYBIND11_DIR})

add_library(allocatorsim SHARED ${C10_CUDA_SIMULATOR_SRCS})
target_include_directories(allocatorsim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${Python_INCLUDE_DIRS}
                            ${PYBIND11_DIR}/include)

install(TARGETS allocatorsim DESTINATION ${TORCH_INSTALL_LIB_DIR})