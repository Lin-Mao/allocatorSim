set(C10_CUDA_SIMULATOR_SRCS
    src/allocator_conf.cpp
    src/allocator_sim.cpp
    src/allocator_mgr.cpp
    src/allocator_prof.cpp
    src/allocator_utils.cpp
)

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

add_library(allocatorsim SHARED ${C10_CUDA_SIMULATOR_SRCS})
target_include_directories(allocatorsim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS allocatorsim DESTINATION "${TORCH_INSTALL_LIB_DIR}")