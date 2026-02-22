// ============================================================================
// Pooling Layer Implementations
// ============================================================================

#include "layers/pooling_layers.h"
#include <algorithm>
#include <cmath>

// Forward declaration of GPU function
void maxpool_forward_gpu(
    const float* d_input, float* d_output,
    int B, int C, int in_h, int in_w,
    int out_h, int out_w,
    int pool_h, int pool_w, int stride);

// ============================================================================
// MaxPooling Layer
// ============================================================================

MaxPoolingLayer::MaxPoolingLayer(int pool_size, int stride)
    : pool_size_(pool_size),
      stride_(stride == -1 ? pool_size : stride)
{

}

void MaxPoolingLayer::forward(const Tensor& input, Tensor& output)
{
    const auto in_shape = input.get_shape(); // [B, C, H, W]
    const int B = in_shape[0];
    const int C = in_shape[1];
    const int H = in_shape[2];
    const int W = in_shape[3];

    const int out_h = (H - pool_size_) / stride_ + 1;
    const int out_w = (W - pool_size_) / stride_ + 1;

    // Check if input is on GPU - if so, use GPU path
    if (input.is_on_gpu()) {
        // Allocate output on GPU if needed
        if (!output.is_on_gpu()) {
            output.allocate_gpu();
        }
        
        // Call GPU kernel
        maxpool_forward_gpu(
            input.gpu_data(), output.gpu_data(),
            B, C, H, W, out_h, out_w,
            pool_size_, pool_size_, stride_
        );
        return;
    }

    // CPU path (original implementation)
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = std::numeric_limits<float>::lowest();
                    for (int kh = 0; kh < pool_size_; ++kh) {
                        for (int kw = 0; kw < pool_size_; ++kw) {
                            const int ih = oh * stride_ + kh;
                            const int iw = ow * stride_ + kw;
                            const float v = input(b, c, ih, iw);
                            if (v > max_val) max_val = v;
                        }
                    }
                    output(b, c, oh, ow) = max_val;
                }
            }
        }
    }
}

std::vector<int> MaxPoolingLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    const int B = input_shape[0];
    const int C = input_shape[1];
    const int H = input_shape[2];
    const int W = input_shape[3];

    const int out_h = (H - pool_size_) / stride_ + 1;
    const int out_w = (W - pool_size_) / stride_ + 1;

    return {B, C, out_h, out_w};
}

// ============================================================================
// Average Pooling Layer
// ============================================================================

AvgPoolingLayer::AvgPoolingLayer(int pool_size, int stride)
    : pool_size_(pool_size),
      stride_(stride == -1 ? pool_size : stride)
{
}

void AvgPoolingLayer::forward(const Tensor& input, Tensor& output)
{
    const auto in_shape = input.get_shape(); // [B, C, H, W]
    const int B = in_shape[0];
    const int C = in_shape[1];
    const int H = in_shape[2];
    const int W = in_shape[3];

    const int out_h = (H - pool_size_) / stride_ + 1;
    const int out_w = (W - pool_size_) / stride_ + 1;

    const float denom = static_cast<float>(pool_size_ * pool_size_);

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < pool_size_; ++kh) {
                        for (int kw = 0; kw < pool_size_; ++kw) {
                            const int ih = oh * stride_ + kh;
                            const int iw = ow * stride_ + kw;
                            sum += input(b, c, ih, iw);
                        }
                    }
                    output(b, c, oh, ow) = sum / denom;
                }
            }
        }
    }
}

std::vector<int> AvgPoolingLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    const int B = input_shape[0];
    const int C = input_shape[1];
    const int H = input_shape[2];
    const int W = input_shape[3];

    const int out_h = (H - pool_size_) / stride_ + 1;
    const int out_w = (W - pool_size_) / stride_ + 1;

    return {B, C, out_h, out_w};
}
