// ============================================================================
// CPU Convolution Layer Implementation
// ============================================================================
// 
// TODO: Implement 2D convolution using nested loops
// 
// Algorithm:
// 1. For each output position (h_out, w_out):
//    2. For each output filter k:
//       3. For each input channel c:
//          4. For each kernel position (kh, kw):
//             5. Load input[b, c, h_in + kh, w_in + kw]
//             6. Multiply with kernel[k, c, kh, kw]
//             7. Add to accumulator
//       8. Add bias[k] and store in output[b, k, h_out, w_out]
// 
// Performance considerations:
// - Use row-major memory layout for cache efficiency
// - Minimize boundary checking in inner loops
// - Consider loop tiling/blocking for better cache locality
//

#include "layers/conv_layer_cpu.h"
#include <cstring>
#include <algorithm>

ConvolutionLayerCPU::ConvolutionLayerCPU(int num_filters, int kernel_h, int kernel_w,
                                         int stride, int padding, int in_channels)
    : num_filters_(num_filters),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_(stride),
      padding_(padding),
      in_channels_(in_channels),
      kernels_(0, 0, 0, 0),  // Will be set by set_weights()
      bias_(num_filters, 0.0f)
{
    // TODO: Initialize bias vector
}

void ConvolutionLayerCPU::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement 2D convolution
    // 
    // Steps:
    // 1. Validate input shape
    // 2. Calculate output dimensions:
    //    out_h = (in_h + 2*padding - kernel_h) / stride + 1
    //    out_w = (in_w + 2*padding - kernel_w) / stride + 1
    // 3. Allocate output tensor if needed
    // 4. For each batch sample b:
    //    For each output filter k:
    //       For each output position (h, w):
    //          For each input channel c:
    //             For each kernel position (kh, kw):
    //                Accumulate: output[b,k,h,w] += input[...] * kernel[...]
    //          Add bias: output[b,k,h,w] += bias[k]
    //
    // Hint: Handle padding by checking boundary conditions
    throw std::runtime_error("Not implemented");
}

std::vector<int> ConvolutionLayerCPU::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Calculate output spatial dimensions
    // input_shape = [batch, in_channels, height, width]
    // 
    // out_h = (height + 2*padding - kernel_h) / stride + 1
    // out_w = (width + 2*padding - kernel_w) / stride + 1
    // 
    // Return [batch, num_filters, out_h, out_w]
    throw std::runtime_error("Not implemented");
}

void ConvolutionLayerCPU::set_weights(const Tensor& kernels)
{
    // TODO: Store kernels (can clone or store reference)
    // kernels shape: [num_filters, in_channels, kernel_h, kernel_w]
}

void ConvolutionLayerCPU::set_bias(const std::vector<float>& bias)
{
    // TODO: Copy bias values
    // bias size: [num_filters]
}
