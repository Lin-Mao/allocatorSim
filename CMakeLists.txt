cmake_minimum_required(VERSION 3.16)

set(C10_CUDA_SIMULATOR_SRCS
    src/allocator_config.cpp
    src/allocator_simulator.cpp
    src/allocator_manager.cpp
    src/allocator_profiler.cpp
    src/allocator_utils.cpp
    src/utils/python_states.cpp
    src/utils/hash.cpp
    src/utils/unwind_utils.cpp
    src/utils/sanitizer_api.cpp
)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)

if(NOT TORCH_INSTALL_LIB_DIR)
    set(TORCH_INSTALL_LIB_DIR lib)
endif()
message(STATUS "TORCH_INSTALL_LIB_DIR: ${TORCH_INSTALL_LIB_DIR}")

# python
find_package(Python3 COMPONENTS Interpreter Development)
if (NOT ${Python_FOUND})
    message(SEND_ERROR "PYTHON not found")
endif()
include_directories(${Python_INCLUDE_DIRS})

# libunwind
set(LIBUNWIND_LIBRARIES /usr/lib/x86_64-linux-gnu)
find_library(LIBUNWIND_LIB NAMES unwind PATHS ${LIBUNWIND_LIBRARIES})
if (NOT LIBUNWIND_LIB)
    message(SEND_ERROR "LIBUNWIND not found")
endif()

# cuda-sanitizer
set(CUDA_DIR /usr/local/cuda)
set(SANITIZER_DIR ${CUDA_DIR}/compute-sanitizer)

find_library(SANITIZER_LIB NAMES sanitizer-public PATHS ${SANITIZER_DIR})
if (NOT SANITIZER_LIB)
    message(SEND_ERROR "SANITIZER not found")
endif()

set(PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
add_subdirectory(${PYBIND11_DIR})

add_library(allocatorsim SHARED ${C10_CUDA_SIMULATOR_SRCS})
target_include_directories(allocatorsim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${Python_INCLUDE_DIRS}
                            ${PYBIND11_DIR}/include
                            ${SANITIZER_DIR}/include
                            ${CUDA_DIR}/include)

target_link_libraries(allocatorsim ${Python_LIBRARIES} ${SANITIZER_LIB} ${LIBUNWIND_LIB})

install(TARGETS allocatorsim DESTINATION ${TORCH_INSTALL_LIB_DIR})