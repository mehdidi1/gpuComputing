#pragma once

#include "../tensor.h"
#include <memory>

/**
 * ============================================================================
 * ILayer - Abstract Base Class for Neural Network Layers
 * ============================================================================
 * 
 * Defines the interface that all layer implementations must follow.
 * Each layer implements the forward pass computation.
 * 
 */
class ILayer {
public:
    virtual ~ILayer() = default;
    
    /**
     * Forward pass: Compute layer output from input
     * 
     * @param input  - Input tensor
     * @param output - Output tensor (should be pre-allocated with correct dimensions)
     * 
     */
    virtual void forward(const Tensor& input, Tensor& output) = 0;
    
    /**
     * Get the output shape given an input shape
     * 
     * Useful for determining intermediate tensor sizes in the network
     * 
     * @param input_shape - [batch, channels, height, width] or [batch, features]
     * @return output_shape - [batch, ..., ..., ...] with layer-specific dimensions
     * 
     */
    virtual std::vector<int> get_output_shape(const std::vector<int>& input_shape) const = 0;
};
