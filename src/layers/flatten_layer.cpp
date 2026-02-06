// ============================================================================
// Flatten Layer Implementation
// ============================================================================
// 
// TODO: Implement reshape from 4D to 2D
// 

#include "layers/flatten_layer.h"
#include <stdexcept>

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

    // Copie en respectant le stockage row-major
    for (int b = 0; b < B; ++b) {
        int index = 0;

        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    output(b, index++) = input(b, c, h, w);
                }
            }
        }
    }
}

std::vector<int> FlattenLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Flatten last 3 dimensions into 1
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
