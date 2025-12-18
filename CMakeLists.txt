cmake_minimum_required(VERSION 3.18)
project(CudaCNN LANGUAGES CXX CUDA)

# ---- 1. Settings & Standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization: Use 'native' to target the GPU currently in your machine
set(CMAKE_CUDA_ARCHITECTURES native)

# ---- 2. Dependencies ----
find_package(CUDAToolkit REQUIRED)

# ---- 3. Create the Library ----
# We combine CPU and GPU sources into one library.
# Objects from .cpp use the host compiler (gcc/msvc), .cu use nvcc.
add_library(cnn_core SHARED
    src/layers/cpu/conv_layer.cpp
    src/layers/cuda/conv_layer.cu
)

# ---- 4. Include & Link ----
target_include_directories(cnn_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(cnn_core PUBLIC 
    CUDA::cudart
    CUDA::cublas  # Useful for the parallel version
)

# ---- 5. Compile Definitions ----
# This allows you to use #ifdef HAS_CUDA in your C++ code
target_compile_definitions(cnn_core PUBLIC HAS_CUDA)

# ---- 6. Main Executable ----
add_executable(cnn_app src/main.cpp)
target_link_libraries(cnn_app PRIVATE cnn_core)
