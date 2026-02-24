# CNN MNIST Inference

GPU-accelerated CNN implementation for MNIST digit classification.

## How to Run

```bash
cd build
./cnn_app [cpu|gpu] [num_samples]
```

**Examples:**
```bash
./cnn_app gpu 1000    # Process 1000 samples on GPU
./cnn_app cpu 100     # Process 100 samples on CPU
./cnn_app             # Default: GPU mode, 100 samples
```

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j4
```

## Key Files & Usage

### Core Classes

**Tensor** (`include/tensor.h`)
- Manages CPU and GPU memory for multi-dimensional arrays
- Supports automatic memory transfers between CPU/GPU

```cpp
// Create tensor [batch=2, channels=3, height=28, width=28]
Tensor input(2, 3, 28, 28);

// Access data and move to GPU
float* data = input.data();
input.to_gpu();
input.to_cpu();

// Clone and query shape
Tensor copy = input.clone();
auto shape = input.get_shape();  // {2, 3, 28, 28}
```

**Model** (`include/model.h`)
- Orchestrates layer execution for forward pass
- Loads weights from binary files

```cpp
// Create and configure model
Model model;
model.add_layer(std::make_unique<ConvolutionLayerGPU>(32, 5, 5, 1, 2, 1));
model.add_layer(std::make_unique<ReLULayer>());
model.add_layer(std::make_unique<MaxPoolingLayer>(2, 2));
model.add_layer(std::make_unique<FullyConnectedLayerGPU>(3136, 128));

// Load pre-trained weights
model.load_weights("model_weights.bin");

// Run inference
Tensor input(batch_size, 1, 28, 28);
Tensor output = model.forward(input);
```

**Layers** (`include/layers/`)
- Convolution, FC, ReLU, MaxPooling, Flatten
- Both CPU and GPU implementations available

```cpp
// GPU convolution: 32 filters, 5x5 kernel, stride=1, padding=2
auto conv = std::make_unique<ConvolutionLayerGPU>(32, 5, 5, 1, 2, 1);

// GPU fully connected: 3136 inputs → 128 outputs
auto fc = std::make_unique<FullyConnectedLayerGPU>(3136, 128);

// CPU versions (for comparison)
auto cpu_conv = std::make_unique<ConvolutionLayerCPU>(32, 5, 5, 1, 2, 1);
auto cpu_fc = std::make_unique<FullyConnectedLayer>(3136, 128);
```

### Example: Custom Inference

See `src/main.cpp` for a complete example:
- Command-line argument parsing (CPU/GPU mode)
- Batch processing of MNIST data
- Performance timing

```cpp
// Load model
Model model;
model.add_layer(...);
model.load_weights("model_weights.bin");

// Process batches
for (int batch = 0; batch < num_batches; batch++) {
    Tensor input = load_batch(data, batch_size);
    input.to_gpu();
    Tensor output = model.forward(input);
    process_results(output);
}
```

## Project Overview

This project implements a CNN for MNIST digit classification with both CPU and GPU versions:
- **Architecture**: 2 conv layers (32, 64 filters) → 2 FC layers (128, 10 units)
- **GPU Optimization**: Shared memory tiling for convolution and FC layers
- **Accuracy**: Verified against PyTorch with 100% prediction match

## Requirements

- CUDA Toolkit 12.0+
- CMake 3.18+
- C++17 compiler

