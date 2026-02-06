#pragma once

#include "tensor.h"
#include <vector>

using Shape = std::vector<int>;

class FlattenLayer {
public:
    // Constructor
    FlattenLayer() = default;

    // Forward pass
    void forward(const Tensor& input, Tensor& output);

    // Output shape computation
    Shape get_output_shape(const Shape& input_shape) const;
};
