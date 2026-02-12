// ============================================================================
// Flatten Layer Implementation
// ============================================================================
// 
// TODO: Implement reshape from 4D to 2D
// 

#include "layers/flatten_layer.h"
#include <cstring>
#include <stdexcept>

FlattenLayer::FlattenLayer() {}

void FlattenLayer::forward(const Tensor& input, Tensor& output)
{
    
    const int B = input.batch();
    const int C = input.channels();
    const int H = input.height();
    const int W = input.width();

    const int flattened_size = C * H * W;

    // Vérifier que output est bien [B, C*H*W]
    if (output.ndims() != 2 ||
        output.batch() != B ||
        output.channels() != flattened_size) {
        throw std::runtime_error("FlattenLayer::forward: output shape mismatch");
    }

    // Copy all data (same total size) in row-major order
    std::memcpy(output.data(), input.data(),
                static_cast<size_t>(B * flattened_size) * sizeof(float));
}

std::vector<int> FlattenLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // input_shape = [batch, channels, height, width]
    // output_shape = [batch, channels*height*width]
    if (input_shape.size() != 4) {
        throw std::runtime_error("FlattenLayer::get_output_shape expects 4D shape");
    }

    int B = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];

    return {B, C * H * W};
}
