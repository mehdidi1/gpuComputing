cmake_minimum_required(VERSION 3.18)
project(CudaCNN LANGUAGES CXX CUDA)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture (or set manually, e.g. 75 for Turing, 89 for Ada)
set(CMAKE_CUDA_ARCHITECTURES native)

# CUDA debug: set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

# Dependencies
find_package(CUDAToolkit REQUIRED)

# Optional: find_package(BLAS REQUIRED)

# CNN Core Library

add_library(cnn_core SHARED
    src/tensor.cpp
    src/layers/cpu/conv_layer.cpp
    src/layers/activation_layers.cpp
    src/layers/pooling_layers.cpp
    src/layers/flatten_layer.cpp
    src/layers/fully_connected_layer.cpp
    src/layers/gpu/conv_layer.cu
    src/layers/gpu/activation_gpu.cu
    src/layers/gpu/pooling_gpu.cu
    src/layers/gpu/fc_layer.cu
    src/model.cpp
)

# Include directories
target_include_directories(cnn_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(cnn_core PUBLIC CUDA::cudart CUDA::cublas)

target_compile_definitions(cnn_core PUBLIC HAS_CUDA)

# Main executable
add_executable(cnn_app src/main.cpp)

target_link_libraries(cnn_app PRIVATE cnn_core)

# Build options
option(ENABLE_GPU "Enable GPU support with CUDA" ON)
if(NOT ENABLE_GPU)
    target_compile_definitions(cnn_core PUBLIC DISABLE_GPU)
endif()


option(ENABLE_DEBUG "Enable debug output and logging" OFF)
if(ENABLE_DEBUG)
    target_compile_definitions(cnn_core PUBLIC DEBUG_OUTPUT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

# Installation (optional)
# install(TARGETS cnn_core cnn_app LIBRARY DESTINATION lib EXECUTABLE DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)
