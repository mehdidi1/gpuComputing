// ============================================================================
// Pooling Layer Implementations
// ============================================================================
// 
// TODO: Implement MaxPooling and AvgPooling
// 

#include "layers/pooling_layers.h"
#include <algorithm>
#include <cmath>

// ============================================================================
// MaxPooling Layer
// ============================================================================

MaxPoolingLayer::MaxPoolingLayer(int pool_size, int stride)
    : pool_size_(pool_size),
      stride_(stride == -1 ? pool_size : stride)
{
    // TODO: Initialize pooling parameters
    // If stride not specified, use pool_size (non-overlapping)
}

void MaxPoolingLayer::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement max pooling
    // 
    // For each output position (h, w):
    //   1. Define input window: input[b,c, h*stride:h*stride+pool_size, 
    //                                      w*stride:w*stride+pool_size]
    //   2. Find maximum value in window
    //   3. Store in output[b,c,h,w]
    // 
    // Note: Batch and channel dimensions unchanged
    throw std::runtime_error("Not implemented");
}

std::vector<int> MaxPoolingLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Calculate pooled spatial dimensions
    // [batch, channels, height, width] -> [batch, channels, out_h, out_w]
    // out_h = (height - pool_size) / stride + 1
    // out_w = (width - pool_size) / stride + 1
    throw std::runtime_error("Not implemented");
}

// ============================================================================
// Average Pooling Layer
// ============================================================================

AvgPoolingLayer::AvgPoolingLayer(int pool_size, int stride)
    : pool_size_(pool_size),
      stride_(stride == -1 ? pool_size : stride)
{
    // TODO: Initialize pooling parameters
}

void AvgPoolingLayer::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement average pooling
    // 
    // Similar to max pooling but:
    //   1. Compute sum of all values in window
    //   2. Divide by pool_size * pool_size to get average
    //   3. Store in output
    throw std::runtime_error("Not implemented");
}

std::vector<int> AvgPoolingLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Same calculation as MaxPoolingLayer
    throw std::runtime_error("Not implemented");
}
