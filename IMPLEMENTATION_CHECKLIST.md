# Implementation Checklist

Track your progress as you implement each component.

## Phase 1: Core Tensor Class

**File:** `src/tensor.cpp` and `include/tensor.h`

### Constructors
- [X] `Tensor(int batch, int channels, int height, int width)` - 4D tensor
- [X] `Tensor(int batch, int features)` - 2D tensor  
- [X] `Tensor(int batch, int channels, int height, int width, const float* data)` - with data
- [X] `~Tensor()` - destructor
- [X] Memory properly allocated with `std::unique_ptr<float[]>`

### Element Access
- [X] `operator()(int b, int c, int h, int w)` - mutable 4D access
- [X] `operator()(int b, int c, int h, int w) const` - const 4D access
- [X] `operator()(int b, int f)` - mutable 2D access
- [X] `operator()(int b, int f) const` - const 2D access
- [O] `compute_index()` helper - efficient linear indexing

### Data Operations
- [X] `fill(float value)` - fill with constant
- [X] `zeros()` - shorthand for fill(0)
- [O] `randn(float mean, float std)` - random normal distribution
- [O] `relu_inplace()` - apply ReLU
- [O] `clone()` - deep copy

### Testing
- [X] Tensor allocation works correctly
- [X] Indexing is row-major and efficient
- [X] Operations produce expected results
- [ ] Memory is properly freed

---

## Phase 2: Activation Layers

**File:** `src/layers/activation_layers.cpp`

### ReLULayer
- [X] Constructor
- [X] `forward(input, output)` - max(0, x)
- [X] `get_output_shape()` - identity
- [X] Test with simple inputs

### SigmoidLayer
- [X] Constructor
- [X] `forward(input, output)` - 1/(1+exp(-x))
- [X] `get_output_shape()` - identity
- [X] Handle numerical stability

### TanhLayer
- [X] Constructor
- [X] `forward(input, output)` - std::tanh
- [X] `get_output_shape()` - identity

### SoftmaxLayer
- [X] Constructor
- [X] `forward(input, output)` - exp(x)/sum(exp(x))
- [X] `get_output_shape()` - identity
- [X] Subtract max for stability
- [X] Handle 2D tensors correctly

### Testing
- [X] Each activation produces correct output
- [X] Output shapes are preserved
- [X] Numerical stability verified
- [X] Compare against manual calculations

---

## Phase 3: Pooling Layers

**File:** `src/layers/pooling_layers.cpp`

### MaxPoolingLayer
- [X] Constructor with pool_size and stride
- [X] `forward(input, output)` - find maximum in windows
- [X] `get_output_shape()` - spatial dimension reduction
- [X] Formula: out_h = (in_h - pool_size) / stride + 1
- [X] Preserve batch and channel dimensions

### AvgPoolingLayer
- [X] Constructor with pool_size and stride
- [X] `forward(input, output)` - compute mean in windows
- [X] `get_output_shape()` - same as max pooling
- [X] Average correctly (divide by pool_size²)

### Testing
- [X] Max/Avg pooling matches manual calculation
- [X] Output dimensions correct
- [X] Works with different pool sizes and strides
- [X] Edge cases handled properly

---

## Phase 4: Simple Layers

**Files:** `src/layers/flatten_layer.cpp` and `src/layers/fully_connected_layer.cpp`

### FlattenLayer
- [ ] Constructor
- [ ] `forward(input, output)` - reshape [B,C,H,W] → [B,C*H*W]
- [ ] `get_output_shape()` - flatten spatial dimensions


### FullyConnectedLayer (CPU)
- [ ] Constructor with in/out features
- [ ] `set_weights(Tensor)` - [in_features, out_features]
- [ ] `set_bias(vector)` - [out_features]
- [ ] `forward(input, output)` - matrix multiply + bias
  - [ ] Use nested loops or BLAS
  - [ ] output = input @ weights + bias

### Testing
- [ ] Flatten preserves data
- [ ] FC output dimensions correct
- [ ] Manual calculation matches
- [ ] Bias addition works
- [ ] Works with various sizes

---

## Phase 5: Convolution (CPU)

**File:** `src/layers/cpu/conv_layer.cpp`

### ConvolutionLayerCPU
- [ ] Constructor with num_filters, kernel_h, kernel_w, stride, padding
- [ ] `set_weights(Tensor)` - [num_filters, in_channels, kernel_h, kernel_w]
- [ ] `set_bias(vector)` - [num_filters]
- [ ] `get_output_shape()` - calculate reduced spatial dims
  - [ ] out_h = (in_h + 2*padding - kernel_h) / stride + 1
  - [ ] out_w = (in_w + 2*padding - kernel_w) / stride + 1
  
### forward() Implementation
- [ ] Iterate through batches
- [ ] Iterate through output filters
- [ ] Iterate through spatial positions (h, w)
- [ ] Iterate through input channels
- [ ] Iterate through kernel positions (kh, kw)
- [ ] Accumulate: output[b,k,h,w] += input[...] * kernel[...]
- [ ] Add bias: output[b,k,h,w] += bias[k]
- [ ] Handle padding by boundary checking

### Testing
- [ ] Output shape matches expected
- [ ] Simple 1x1 kernel works
- [ ] Identity kernel produces correct results
- [ ] Padding works properly
- [ ] Stride works properly
- [ ] Compare small test against manual calculation

### Optimization (Optional)
- [ ] Use BLAS for better performance
- [ ] Implement im2col for efficiency
- [ ] Cache-friendly loop ordering

---

## Phase 6: Model Orchestration

**File:** `src/model.cpp`

### Model Class
- [ ] `add_layer(unique_ptr<ILayer>)` - add to sequence
- [ ] `forward(input)` - execute all layers
  - [ ] Start with input tensor
  - [ ] For each layer:
    - [ ] Get output shape
    - [ ] Allocate output tensor
    - [ ] Call layer.forward()
    - [ ] Set output as input for next layer
  - [ ] Return final output
  
- [ ] `load_weights(filename)` - binary file I/O
  - [ ] Design binary format
  - [ ] Write/read header (magic, version, num_layers)
  - [ ] For each layer:
    - [ ] Write layer type
    - [ ] Write weights dimensions and data
    - [ ] Write bias dimensions and data
    
- [ ] `save_weights(filename)` - complement of load_weights

### Testing
- [ ] Layers execute in correct order
- [ ] Output shapes propagate correctly
- [ ] Weights load and work properly
- [ ] Save/load roundtrip preserves values

---

## Phase 7: PyTorch Integration

**Files:** `python/export_pytorch_model.py` and `python/validate_pytorch_model.py`

### Model Definition
- [ ] SimpleCNN class defined with proper architecture
- [ ] Conv2d, ReLU, MaxPool2d, Linear layers

### Weight Export (`export_pytorch_model.py`)
- [ ] `WeightExporter.export_model()` implementation
  - [ ] Write magic number and version
  - [ ] Iterate through PyTorch layers
  - [ ] Extract weights and biases
  - [ ] Write to binary file with format specification
  
- [ ] Data loading for training
- [ ] Model training function
- [ ] Test data export function

### Model Validation (`validate_pytorch_model.py`)
- [ ] `load_binary_output()` - read C++ results
- [ ] `load_pytorch_output()` - run PyTorch inference
- [ ] `compute_metrics()` - L2 error, max error, relative error
- [ ] `print_metrics()` - formatted output

### Testing
- [ ] PyTorch model can export weights
- [ ] C++ can load exported weights
- [ ] Outputs match within tolerance (1e-4)
- [ ] Different input sizes work

---

## Phase 8: GPU Implementation (CUDA)

**File:** `src/layers/gpu/conv_layer.cu`

### ConvolutionLayerGPU
- [ ] Constructor same as CPU version
- [ ] Destructor with cudaFree for GPU memory
- [ ] `set_weights()` - copy to GPU via cudaMalloc and cudaMemcpy
- [ ] `set_bias()` - copy to GPU

### CUDA Kernel Implementation
- [ ] `__global__ void conv2d_kernel(...)` signature
- [ ] Grid/block configuration
  - [ ] Grid for batch and filters
  - [ ] Block for spatial output positions
  
- [ ] Each thread computes one output element:
  - [ ] Calculate input window position
  - [ ] Load from global memory
  - [ ] Accumulate convolution
  - [ ] Handle padding boundary conditions
  - [ ] Write output
  
- [ ] Alternative optimizations:
  - [ ] Shared memory for kernels
  - [ ] Coalesced memory access
  - [ ] Warp-level operations

### forward() Implementation
- [ ] Copy input to GPU (cudaMemcpy)
- [ ] Launch kernel with proper grid/block
- [ ] Synchronize (cudaDeviceSynchronize)
- [ ] Check for errors (cudaGetLastError)
- [ ] Copy output back to CPU

### Testing
- [ ] GPU output matches CPU output (within tolerance)
- [ ] Different input sizes work
- [ ] Memory is properly allocated/freed
- [ ] No CUDA errors reported

---

## Phase 9: GPU Fully Connected Layer

**File:** `src/layers/fully_connected_layer.cpp` (GPU section)

### FullyConnectedLayerGPU
- [ ] Constructor
- [ ] Destructor with cudaFree
- [ ] GPU memory management (d_weights_, d_bias_)

### forward() with cuBLAS
- [ ] Create cuBLAS handle if needed
- [ ] Copy input to GPU
- [ ] Call cublasSgemm for matrix multiply
  - [ ] Correct dimension mapping
  - [ ] Proper leading dimensions
  - [ ] alpha=1.0, beta=0.0
  
- [ ] Add bias (separate kernel or cuBLAS)
- [ ] Copy output back to CPU
- [ ] Check for errors

### Testing
- [ ] Output matches CPU version
- [ ] Different batch sizes work
- [ ] Different feature dimensions work

---

## Phase 10: Integration & Benchmarking

**File:** `src/main.cpp`

### Complete Example Program
- [ ] Build a CNN model (CPU version)
- [ ] Load weights from file or initialize
- [ ] Create test input
- [ ] Run forward pass and time it
- [ ] Print results

### GPU Benchmarking
- [ ] Replace CPU layers with GPU layers
- [ ] Compare timing: CPU vs GPU vs PyTorch
- [ ] Report speedup factors

### Validation
- [ ] Load PyTorch reference output
- [ ] Compare C++ CPU output
- [ ] Compare C++ GPU output
- [ ] Report error metrics

---

## Phase 11: Testing & Documentation

### Unit Tests
- [ ] test_tensor.cpp - Tensor operations
- [ ] test_layers_cpu.cpp - Each CPU layer
- [ ] test_layers_gpu.cpp - GPU layers vs CPU
- [ ] test_pytorch_compat.cpp - PyTorch comparison

### Integration Tests
- [ ] Full forward pass
- [ ] Various input sizes
- [ ] Different architectures
- [ ] Batch processing

### Benchmarking
- [ ] CPU single-threaded timing
- [ ] GPU timing
- [ ] PyTorch timing
- [ ] Compute speedup factors

### Documentation
- [ ] Code comments explaining algorithms
- [ ] Performance notes
- [ ] Known limitations
- [ ] Future optimization ideas

---

## Verification Checklist

Before claiming implementation is complete:

### Correctness
- [ ] CPU outputs match PyTorch (L2 error < 1e-4)
- [ ] GPU outputs match CPU (L2 error < 1e-3)
- [ ] Works with different input sizes
- [ ] Handles edge cases (stride, padding, etc.)

### Performance
- [ ] GPU at least 5x faster than CPU
- [ ] Scales well with batch size
- [ ] Memory efficient (no leaks)
- [ ] Reasonable latency for inference

### Code Quality
- [ ] No compiler warnings
- [ ] Proper error handling
- [ ] Memory properly managed
- [ ] Well-commented code
- [ ] Consistent naming conventions

### Completeness
- [ ] All required functions implemented
- [ ] All layers working correctly
- [ ] PyTorch integration complete
- [ ] Documentation written
- [ ] Tests passing

---

## Timeline Estimates

| Phase | Duration | Status |
|-------|----------|--------|
| 1. Tensor | 2-3 hours | ☐ |
| 2. Activations | 1 hour | ☐ |
| 3. Pooling | 1 hour | ☐ |
| 4. Flatten+FC | 1-2 hours | ☐ |
| 5. Conv(CPU) | 2-3 hours | ☐ |
| 6. Model | 1 hour | ☐ |
| 7. PyTorch | 2-3 hours | ☐ |
| 8. Conv(GPU) | 3-4 hours | ☐ |
| 9. FC(GPU) | 1-2 hours | ☐ |
| 10. Integration | 1 hour | ☐ |
| 11. Testing | 2-3 hours | ☐ |
| **Total** | **~20 hours** | ☐ |

---

## Tips for Success

✅ **Do:**
- Test incrementally (build and run after each function)
- Compare with PyTorch early and often
- Start simple (test with tiny tensors)
- Use print statements for debugging
- Commit to version control frequently

❌ **Don't:**
- Try to implement everything at once
- Assume your code works without testing
- Ignore numerical stability issues
- Forget about memory management
- Skip the GPU implementation (it's the point!)

---

Good luck! 🚀
