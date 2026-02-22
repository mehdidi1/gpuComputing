// ============================================================================
// Flatten Layer Implementation
// ============================================================================
// 
// Implements reshape from 4D to 2D
// 

#include "layers/flatten_layer.h"
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>

FlattenLayer::FlattenLayer() {}

void FlattenLayer::forward(const Tensor& input, Tensor& output)
{
    
    const int B = input.batch();
    const int C = input.channels();
    const int H = input.height();
    const int W = input.width();

    const int flattened_size = C * H * W;

    // If input is on GPU, just copy GPU pointer (no data movement needed - same layout)
    if (input.is_on_gpu()) {
        if (!output.is_on_gpu()) {
            output.allocate_gpu();
        }
        // Direct copy on GPU (same memory layout)
        cudaMemcpy(output.gpu_data(), input.gpu_data(),
                   B * flattened_size * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        return;
    }

    // CPU path: Copy all data (same total size) in row-major order
    std::memcpy(output.data(), input.data(),
                static_cast<size_t>(B * flattened_size) * sizeof(float));
}

std::vector<int> FlattenLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // input_shape = [batch, channels, height, width]
    // output_shape = [batch, channels*height*width]
    int B = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];

    return {B, C * H * W};
}
