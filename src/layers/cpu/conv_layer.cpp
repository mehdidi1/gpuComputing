// ============================================================================
// CPU Convolution Layer Implementation
// ============================================================================
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
      kernels_(0, 0, 0, 0),
      bias_(num_filters, 0.0f)
{
    std::fill(bias_.begin(), bias_.end(), 0.0f);   //Initialize bias vector
}

void ConvolutionLayerCPU::forward(const Tensor& input, Tensor& output)
{
    // Validate input shape
    const auto in_shape = input.get_shape();
    if (in_shape.size() != 4) {
        throw std::runtime_error("ConvolutionLayerCPU::forward expects 4D input");
    }
    
    const int B = in_shape[0];
    const int C = in_shape[1];
    const int H = in_shape[2];
    const int W = in_shape[3];
    
    if (C != in_channels_) {
        throw std::runtime_error("Input channels mismatch");
    }
    
    // Calculate output dimensions
    const int out_h = (H + 2 * padding_ - kernel_h_) / stride_ + 1;
    const int out_w = (W + 2 * padding_ - kernel_w_) / stride_ + 1;
    
    // Check output tensor size
    const auto out_shape = output.get_shape();
    if (out_shape.size() != 4 || out_shape[0] != B || out_shape[1] != num_filters_ ||
        out_shape[2] != out_h || out_shape[3] != out_w) {
        throw std::runtime_error("Output tensor has incorrect shape.");
    }
    
    // Main convolution loop
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < num_filters_; ++k) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias_[k];  // Start with bias
                    
                    // Convolve with all input channels
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < kernel_h_; ++kh) {
                            for (int kw = 0; kw < kernel_w_; ++kw) {
                                // Calculate input position
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                
                                // Check bounds (padding region is 0)
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float in_val = input(b, c, ih, iw);
                                    float kernel_val = kernels_(k, c, kh, kw);
                                    sum += in_val * kernel_val;
                                }
                            }
                        }
                    }
                    
                    output(b, k, oh, ow) = sum;
                }
            }
        }
    }
}

std::vector<int> ConvolutionLayerCPU::get_output_shape(const std::vector<int>& input_shape) const
{
    if (input_shape.size() != 4) {
        throw std::runtime_error("ConvolutionLayerCPU::get_output_shape expects 4D input");
    }
    
    const int B = input_shape[0];
    const int H = input_shape[2];
    const int W = input_shape[3];
    
    // Calculate output spatial dimensions
    const int out_h = (H + 2 * padding_ - kernel_h_) / stride_ + 1;
    const int out_w = (W + 2 * padding_ - kernel_w_) / stride_ + 1;
    
    return {B, num_filters_, out_h, out_w};
}

void ConvolutionLayerCPU::set_weights(const Tensor& kernels)
{
    // Validate shape
    const auto k_shape = kernels.get_shape();
    if (k_shape.size() != 4 || k_shape[0] != num_filters_ || 
        k_shape[1] != in_channels_ || k_shape[2] != kernel_h_ || k_shape[3] != kernel_w_) {
        throw std::runtime_error("Kernel shape mismatch");
    }
    
    // Deep copy kernels
    kernels_ = kernels.clone();
}

void ConvolutionLayerCPU::set_bias(const std::vector<float>& bias)
{
    // Validate bias size
    if (bias.size() != num_filters_) {
        throw std::runtime_error("Bias size mismatch: expected " + 
                                 std::to_string(num_filters_) + 
                                 " but got " + std::to_string(bias.size()));
    }
    
    // Copy bias values
    bias_ = bias;
}
