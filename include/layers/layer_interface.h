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
 * Layer types to implement:
 * - ConvolutionLayerCPU / ConvolutionLayerGPU
 * - ActivationLayer (ReLU, Sigmoid, etc.)
 * - PoolingLayer (MaxPool, AvgPool)
 * - FlattenLayer
 * - FullyConnectedLayer
 * - SoftmaxLayer
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
     * TODO: Implement in derived classes
     * Each layer should transform input -> output according to its computation
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
     * TODO: Implement in derived classes
     * Example: Conv layer with padding=1, stride=1 keeps spatial dims
     *          MaxPool with pool_size=2, stride=2 halves spatial dims
     */
    virtual std::vector<int> get_output_shape(const std::vector<int>& input_shape) const = 0;
};
