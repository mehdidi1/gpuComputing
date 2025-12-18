# Create a library for CNN components
add_library(layers_lib
    conv_layer.cu
    pooling_layer.cu
)

# Tell CMake this library uses CUDA and cuDNN
target_link_libraries(layers_lib 
    PRIVATE 
        CUDA::cudart
        CUDA::cublas
        CUDA::cudnn
)

# Handle include directories
target_include_directories(layers_lib 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Set Separation Compilation if your kernels call each other across files
set_target_properties(layers_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
