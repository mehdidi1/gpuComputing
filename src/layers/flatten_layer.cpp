// ============================================================================
// Flatten Layer Implementation
// ============================================================================
// 
// TODO: Implement reshape from 4D to 2D
// 

#include "layers/flatten_layer.h"

FlattenLayer::FlattenLayer() {}

void FlattenLayer::forward(const Tensor& input, Tensor& output)
{
    // TODO: Reshape 4D to 2D
    // 
    // [batch, channels, height, width] -> [batch, channels*height*width]
    // 
    // Important: Share the same underlying memory (no data copy)
    // The output tensor should point to the same data as input
    // but with different dimension interpretation
    throw std::runtime_error("Not implemented");
}

std::vector<int> FlattenLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Flatten last 3 dimensions into 1
    // input_shape = [batch, channels, height, width]
    // output_shape = [batch, channels*height*width]
    throw std::runtime_error("Not implemented");
}
