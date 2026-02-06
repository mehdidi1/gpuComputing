# GPU Computing CNN Framework - Implementation Guide

## Overview

This is a complete skeleton for implementing a CNN framework with both CPU and GPU (CUDA) support, designed for a GPU Computing class project. The goal is to:

1. **Implement a CNN framework from scratch** in C++ and CUDA
2. **Compare performance and correctness** with PyTorch
3. **Learn about parallelization** by implementing GPU kernels

## Project Structure

```
gpuComputing/
├── include/
│   ├── tensor.h                           # Tensor class (multi-dimensional arrays)
│   ├── model.h                            # Model class (layer orchestration)
│   └── layers/
│       ├── layer_interface.h              # Base class for all layers
│       ├── conv_layer_cpu.h               # CPU convolution
│       ├── conv_layer_gpu.h               # GPU convolution (CUDA)
│       ├── activation_layers.h            # ReLU, Sigmoid, Tanh, Softmax
│       ├── pooling_layers.h               # MaxPool, AvgPool
│       ├── flatten_layer.h                # Reshape 4D->2D
│       └── fully_connected_layer.h        # Dense/FC layers
│
├── src/
│   ├── tensor.cpp                         # Tensor implementation
│   ├── model.cpp                          # Model implementation
│   ├── main.cpp                           # Main application
│   └── layers/
│       ├── cpu/conv_layer.cpp             # CPU conv implementation
│       ├── gpu/conv_layer.cu              # GPU conv kernel (CUDA)
│       ├── activation_layers.cpp          # Activation implementations
│       ├── pooling_layers.cpp             # Pooling implementations
│       ├── flatten_layer.cpp              # Flatten implementation
│       └── fully_connected_layer.cpp      # FC implementation
│
├── python/
│   ├── export_pytorch_model.py            # Export PyTorch models to binary format
│   └── validate_pytorch_model.py          # Validate C++ against PyTorch
│
├── tests/                                 # Test cases (to be implemented)
│
└── CMakeLists.txt                         # Build configuration
```

## Quick Start

### 1. Build the Project

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Build options:**
```bash
# Enable debug output
cmake .. -DENABLE_DEBUG=ON

# With custom CUDA architecture (e.g., for RTX 2080)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75

# Disable GPU support (CPU-only)
cmake .. -DENABLE_GPU=OFF
```

### 2. Run the Application

```bash
./cnn_app
```

## Implementation Guide

### Phase 1: Core Data Structures

**File:** `src/tensor.cpp`

TODO items:
- [ ] Implement `Tensor` constructors with proper memory allocation
- [ ] Implement element access operators `operator()(int, int, int, int)` with efficient indexing
- [ ] Implement `fill()`, `zeros()`, `randn()` utility functions
- [ ] Implement `relu_inplace()` activation

**Key concepts:**
- Row-major (C-style) memory layout for cache efficiency
- Linear indexing formula: `index = b*(C*H*W) + c*(H*W) + h*W + w`

### Phase 2: CPU Layer Implementations

Implement each layer in the following order (easier to harder):

#### 2.1 Activation Layers
**File:** `src/layers/activation_layers.cpp`

- [ ] ReLU: `max(0, x)`
- [ ] Sigmoid: `1 / (1 + exp(-x))` (watch for numerical stability)
- [ ] Tanh: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
- [ ] Softmax: `exp(x) / sum(exp(x))` (subtract max for stability)

#### 2.2 Pooling Layers
**File:** `src/layers/pooling_layers.cpp`

- [ ] MaxPooling: For each window, output maximum value
- [ ] AvgPooling: For each window, output average value

**Formula:** `out_h = (in_h - pool_size) / stride + 1`

#### 2.3 Flatten Layer
**File:** `src/layers/flatten_layer.cpp`

- [ ] Reshape `[batch, C, H, W]` → `[batch, C*H*W]`
- [ ] Share memory (no data copy)

#### 2.4 Fully Connected Layer
**File:** `src/layers/fully_connected_layer.cpp`

- [ ] Matrix multiplication: `output = input @ weights + bias`
- [ ] Dimension: `[batch, in_features] @ [in_features, out_features] → [batch, out_features]`
- [ ] Optimization: Use nested loops or BLAS library

#### 2.5 Convolution Layer (CPU)
**File:** `src/layers/cpu/conv_layer.cpp`

- [ ] Implement 2D convolution with nested loops
- [ ] Handle padding and stride parameters
- [ ] Calculate output dimensions: `out_h = (in_h + 2*padding - kernel_h) / stride + 1`

### Phase 3: Model Orchestration

**File:** `src/model.cpp`

- [ ] Implement `add_layer()` to build sequential model
- [ ] Implement `forward()` to execute layers in sequence
- [ ] Implement `load_weights()` and `save_weights()` for binary file I/O

### Phase 4: GPU Implementation (CUDA)

#### 4.1 GPU Convolution
**File:** `src/layers/gpu/conv_layer.cu`

- [ ] Write CUDA kernel `__global__ void conv2d_kernel(...)`
- [ ] Implement grid/block configuration
- [ ] Each thread computes one output element
- [ ] Handle memory transfers: `cudaMemcpy()`
- [ ] Add error checking: `cudaGetLastError()`

**Optimization opportunities:**
- Shared memory for kernel data caching
- Coalesced memory access
- Texture memory for inputs
- `float4` loads for bandwidth

#### 4.2 GPU Fully Connected Layer
**File:** `src/layers/fully_connected_layer.cpp`

- [ ] Use cuBLAS `cublasSgemm()` for efficient matrix multiply
- [ ] Implement bias addition with a simple kernel

### Phase 5: PyTorch Integration

#### 5.1 Model Exporter
**File:** `python/export_pytorch_model.py`

- [ ] Define PyTorch CNN model (`SimpleCNN` class)
- [ ] Implement `WeightExporter.export_model()` to save binary format
- [ ] Implement `load_mnist_data()` for training data
- [ ] Implement `train_model()` for model training
- [ ] Implement `export_test_data()` for validation

#### 5.2 Model Validator
**File:** `python/validate_pytorch_model.py`

- [ ] Implement `load_binary_output()` to read C++ inference results
- [ ] Implement `compute_metrics()` for error analysis:
  - L2 error
  - Max error
  - Relative error
  - Cosine similarity
- [ ] Implement comparison tolerance checking

### Phase 6: Testing and Validation

Create tests in `tests/` directory:

- [ ] `test_tensor.cpp` - Test Tensor allocation, indexing, operations
- [ ] `test_layers_cpu.cpp` - Test each CPU layer against simple inputs
- [ ] `test_layers_gpu.cpp` - Test GPU layers against CPU versions
- [ ] `test_pytorch_compatibility.cpp` - Load weights, compare with PyTorch

**Testing strategy:**
1. Hardcode simple test inputs
2. Compare CPU output against manual calculation
3. Compare GPU output against CPU output
4. Compare against PyTorch reference

## Implementation Strategies

### CPU Optimization Tips

1. **Cache efficiency:**
   - Access data in row-major order
   - Process blocks of data that fit in cache
   - Minimize memory bandwidth

2. **Loop optimization:**
   - Loop tiling for better data locality
   - Unroll inner loops
   - Minimize function calls in hot loops

3. **BLAS usage:**
   - Link against optimized BLAS (OpenBLAS, MKL)
   - Use `cblas_sgemm()` for matrix multiply
   - Reduce custom loop implementations

### GPU Optimization Tips

1. **Kernel design:**
   - Use 2D/3D blocks for spatial problems
   - Map one output element per thread initially
   - Later optimize for multiple elements per thread

2. **Memory hierarchy:**
   - Copy hot data to shared memory
   - Use constant memory for kernels/parameters
   - Coalesce global memory accesses

3. **Synchronization:**
   - Use `__syncthreads()` within blocks
   - Minimize synchronization overhead
   - Use warp-level operations when possible

4. **cuBLAS usage:**
   - Use `cublasSgemm()` for fully connected layers
   - Use `cublasScopy()` for bias addition
   - Profile to identify bottlenecks

## Debugging Tips

### CPU Debugging
```cpp
// Add debug output
#ifdef DEBUG_OUTPUT
    std::cout << "Input shape: [" << input.batch() << "," 
              << input.channels() << "," << input.height() << "," 
              << input.width() << "]" << std::endl;
#endif

// Check dimensions
assert(output.size() > 0);
```

### GPU Debugging
```cuda
// Check CUDA errors
cudaError_t err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
}

// Synchronize and check kernel errors
cudaDeviceSynchronize();
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
}
```

### Compare Outputs
```python
# Python validation
import numpy as np
cpp_out = np.load("cpp_output.npy")
pytorch_out = np.load("pytorch_output.npy")
error = np.abs(cpp_out - pytorch_out)
print(f"Max error: {error.max()}")
print(f"Mean error: {error.mean()}")
print(f"Relative error: {error.max() / abs(pytorch_out).max()}")
```

## Performance Benchmarking

Once implementation is complete:

```cpp
#include <chrono>

// Time a forward pass
auto start = std::chrono::high_resolution_clock::now();
Tensor output = model.forward(input);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

std::cout << "Forward pass time: " << duration.count() << " ms" << std::endl;
```

Compare:
- CPU single-threaded
- CPU with OpenMP parallelization
- GPU CUDA implementation
- PyTorch (CPU and GPU)

## Common Issues

### Build Errors

**Missing CUDA headers:**
```bash
# On Ubuntu:
sudo apt-get install nvidia-cuda-toolkit

# Verify CUDA is installed:
nvcc --version
```

**CMake can't find CUDA:**
```bash
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

### Runtime Errors

**Segmentation fault:**
- Check array bounds in Tensor indexing
- Verify pointer validity in GPU code
- Use `valgrind` or CUDA debugger (`cuda-gdb`)

**Incorrect results:**
- Compare layer outputs with NumPy/PyTorch
- Check dimension calculations
- Verify memory layout (row-major vs column-major)

## Deliverables Checklist

- [ ] Complete Tensor implementation
- [ ] Implement all CPU layers
- [ ] Implement PyTorch exporter
- [ ] Validate CPU output against PyTorch
- [ ] Implement GPU convolution kernel
- [ ] Implement GPU fully connected layer
- [ ] Validate GPU output against CPU
- [ ] Performance benchmarks (CPU vs GPU vs PyTorch)
- [ ] Documentation/Report

## References

### CUDA
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUDA Samples](https://github.com/nvidia/cuda-samples)

### PyTorch
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Model Saving/Loading](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

### CNNs
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/)
- [Understanding Convolutions](https://colah.github.io/posts/2014-07-Understanding-Convolutions/)

## Questions & Support

- Check TODO comments in code for hints
- Review function prototypes and docstrings
- Ask instructor for clarification on algorithms
- Refer to PyTorch source code for reference implementations
