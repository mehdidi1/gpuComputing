cmake_minimum_required(VERSION 3.18)
project(CudaCNN LANGUAGES CXX CUDA)

# ============================================================================
# CMAKE BUILD CONFIGURATION
# ============================================================================

# ---- 1. Compiler Settings & C++ Standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")  # Optimization

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU architecture on the current machine
# Set to specific arch if needed: set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 2060, RTX 2070, etc.
# Options: 50 (Maxwell), 61 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 89 (Ada)
set(CMAKE_CUDA_ARCHITECTURES native)

# CUDA debug symbols (uncomment for debugging)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

# ---- 2. Find Required Dependencies ----
find_package(CUDAToolkit REQUIRED)

# Optional: Find BLAS for optimized linear algebra on CPU
# find_package(BLAS REQUIRED)

# ============================================================================
# CREATE CNN CORE LIBRARY
# ============================================================================
# This combines CPU implementations (.cpp) and GPU implementations (.cu)
# into a single shared library for easy integration

add_library(cnn_core SHARED
    # Tensor implementation
    src/tensor.cpp
    
    # CPU layer implementations
    src/layers/cpu/conv_layer.cpp
    src/layers/activation_layers.cpp
    src/layers/pooling_layers.cpp
    src/layers/flatten_layer.cpp
    src/layers/fully_connected_layer.cpp
    
    # GPU layer implementations (CUDA)
    src/layers/gpu/conv_layer.cu
    
    # Model orchestration
    src/model.cpp
)

# ---- 3. Include Directories ----
# Make all header files available to this library and its users
target_include_directories(cnn_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ---- 4. Link Against Dependencies ----
# CUDA Runtime - Required for all CUDA code
target_link_libraries(cnn_core PUBLIC 
    CUDA::cudart
)

# cuBLAS - For optimized matrix operations (matrix multiply, etc.)
# Used by FC layers for efficient GEMM (General Matrix-Matrix Multiply)
target_link_libraries(cnn_core PUBLIC 
    CUDA::cublas
)

# Optional: Link BLAS for CPU matrix operations
# if(BLAS_FOUND)
#     target_link_libraries(cnn_core PUBLIC ${BLAS_LIBRARIES})
# endif()

# ---- 5. Compile Definitions ----
# This preprocessor define allows conditional compilation:
# #ifdef HAS_CUDA
#     // GPU code here
# #endif
target_compile_definitions(cnn_core PUBLIC HAS_CUDA)

# ============================================================================
# CREATE EXECUTABLE (Main Application)
# ============================================================================
# Links against cnn_core library
add_executable(cnn_app 
    src/main.cpp
)

target_link_libraries(cnn_app PRIVATE cnn_core)

# ============================================================================
# BUILD OPTIONS (Optional, for easy configuration)
# ============================================================================

# Option to enable/disable GPU support
option(ENABLE_GPU "Enable GPU support with CUDA" ON)
if(NOT ENABLE_GPU)
    target_compile_definitions(cnn_core PUBLIC DISABLE_GPU)
endif()

# Option for debug output
option(ENABLE_DEBUG "Enable debug output and logging" OFF)
if(ENABLE_DEBUG)
    target_compile_definitions(cnn_core PUBLIC DEBUG_OUTPUT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")  # Add debug symbols
endif()

# ==========================================================================
# TESTING
# ==========================================================================

enable_testing()

# Tensor unit tests
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE cnn_core)
add_test(NAME TensorTest COMMAND test_tensor)

# Pooling layer unit tests
add_executable(test_pooling tests/test_pooling.cpp)
target_link_libraries(test_pooling PRIVATE cnn_core)
add_test(NAME PoolingTest COMMAND test_pooling)

# Activation layer unit tests
add_executable(test_activation tests/test_activation.cpp)
target_link_libraries(test_activation PRIVATE cnn_core)
add_test(NAME ActivationTest COMMAND test_activation)

# Additional tests can be added later
# add_executable(test_layers tests/test_layers.cpp)
# target_link_libraries(test_layers PRIVATE cnn_core)
# add_test(NAME LayerTest COMMAND test_layers)

# ============================================================================
# INSTALLATION (Optional)
# ============================================================================
# Uncomment to enable installation targets

# install(TARGETS cnn_core cnn_app
#         LIBRARY DESTINATION lib
#         EXECUTABLE DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)
