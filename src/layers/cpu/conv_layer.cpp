// ============================================================================
// CPU Convolution Layer Implementation
// ============================================================================
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
      kernels_(0, 0, 0, 0),
      bias_(num_filters, 0.0f)
{
    std::fill(bias_.begin(), bias_.end(), 0.0f);   //Initialize bias vector
}

void ConvolutionLayerCPU::forward(const Tensor& input, Tensor& output)
{
    // Validate input shape
    const auto in_shape = input.get_shape();
    
    const int B = in_shape[0];
    const int C = in_channels_;
    const int H = in_shape[2];
    const int W = in_shape[3];
    
    // Calculate output dimensions
    const int out_h = (H + 2 * padding_ - kernel_h_) / stride_ + 1;
    const int out_w = (W + 2 * padding_ - kernel_w_) / stride_ + 1;
    
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
    // Deep copy kernels
    kernels_ = kernels.clone();
}

void ConvolutionLayerCPU::set_bias(const std::vector<float>& bias)
{
    // Copy bias values
    bias_ = bias;
}
