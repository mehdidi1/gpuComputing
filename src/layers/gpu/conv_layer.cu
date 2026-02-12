// ============================================================================
// GPU Convolution Layer Implementation (CUDA)
// ============================================================================
// 
// TODO: Implement CUDA kernel for 2D convolution
// 
// Parallelization strategy:
// - Grid dimension: 2D (num_filters, batch_size) to map output feature maps
// - Block dimension: 2D (32x8 or similar) to map spatial output positions
// - Each thread computes one output element
// 
// CUDA kernel structure:
// 
// __global__ void conv2d_kernel(const float* input,   // [B, C, H, W]
//                                const float* kernels, // [K, C, KH, KW]
//                                const float* bias,    // [K]
//                                float* output,        // [B, K, H', W']
//                                int in_h, int in_w, int in_c,
//                                int out_h, int out_w,
//                                int kernel_h, int kernel_w,
//                                int stride, int padding)
// {
//     int batch = blockIdx.y;
//     int filter = blockIdx.x;
//     int h_out = threadIdx.y + blockDim.y * blockIdx.z;
//     int w_out = threadIdx.x + blockDim.x * blockIdx.w;
//     
//     if (h_out >= out_h || w_out >= out_w) return;
//     
//     float sum = bias[filter];
//     for (int c = 0; c < in_c; ++c) {
//         for (int kh = 0; kh < kernel_h; ++kh) {
//             for (int kw = 0; kw < kernel_w; ++kw) {
//                 int h_in = h_out * stride + kh - padding;
//                 int w_in = w_out * stride + kw - padding;
//                 if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
//                     // Compute linear indices and do the multiplication
//                     // sum += input[...] * kernels[...]
//                 }
//             }
//         }
//     }
//     output[...] = sum;
// }
// 
// TODO: Consider optimizations:
// - Use shared memory for kernel data (if it fits)
// - Use texture memory for input (if available)
// - Coalesce memory accesses (consecutive threads access consecutive memory)
// - Use float4 for loading data (loads 4 floats in one transaction)
//

#include "layers/conv_layer_gpu.h"

// ============================================================================
// CUDA Kernel Declaration
// ============================================================================

// TODO: Declare CUDA kernel
// __global__ void conv2d_kernel(const float* input, const float* kernels,
//                                const float* bias, float* output, ...);

// ============================================================================
// ConvolutionLayerGPU Implementation
// ============================================================================

ConvolutionLayerGPU::ConvolutionLayerGPU(int num_filters, int kernel_h, int kernel_w,
                                         int stride, int padding, int in_channels)
    : num_filters_(num_filters),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_(stride),
      padding_(padding),
      in_channels_(in_channels),
      d_kernels_(nullptr),
      d_bias_(nullptr),
      kernels_(0, 0, 0, 0),
      bias_(num_filters, 0.0f)
{
    // TODO: Initialize GPU pointers to nullptr
}

ConvolutionLayerGPU::~ConvolutionLayerGPU()
{
    // TODO: Free GPU memory
    // if (d_kernels_) cudaFree(d_kernels_);
    // if (d_bias_) cudaFree(d_bias_);
}

void ConvolutionLayerGPU::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement GPU convolution
    // 
    // Steps:
    // 1. Allocate GPU memory for input, output, kernels, bias if not done
    // 2. Copy input data to GPU: cudaMemcpy(d_input, input.data(), ...)
    // 3. Launch kernel: 
    //    dim3 grid(...);
    //    dim3 block(...);
    //    conv2d_kernel<<<grid, block>>>(d_input, d_kernels, d_bias, d_output, ...);
    // 4. Copy output back to CPU: cudaMemcpy(output.data(), d_output, ...)
    // 5. Check for errors: cudaGetLastError()
    throw std::runtime_error("Not implemented");
}

std::vector<int> ConvolutionLayerGPU::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Same as CPU version
    // Calculate output spatial dimensions based on conv parameters
    throw std::runtime_error("Not implemented");
}

void ConvolutionLayerGPU::set_weights(const Tensor& kernels)
{
    // TODO: Copy kernels to GPU memory
    // 1. Allocate GPU memory if not already done
    // 2. Copy data from CPU to GPU using cudaMemcpy
    // 3. Store CPU backup in kernels_
}

void ConvolutionLayerGPU::set_bias(const std::vector<float>& bias)
{
    // TODO: Copy bias to GPU memory
    // 1. Allocate GPU memory if not already done
    // 2. Copy data using cudaMemcpy
    // 3. Store CPU backup in bias_
}
