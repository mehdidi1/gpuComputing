#include "layers/activation_layers.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ReLU kernel
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// GPU-accelerated ReLU forward
void relu_forward_gpu(const float* d_input, float* d_output, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
}
