#pragma once

#include "../tensor.h"
#include "layer_interface.h"

/**
 * ============================================================================
 * Flatten Layer - Reshape 4D tensor to 2D
 * ============================================================================
 * 
 * Converts [batch, channels, height, width] to [batch, channels*height*width]
 * 
 * This is typically the transition from convolutional layers to fully-connected layers
 * 
 */
class FlattenLayer : public ILayer {
public:
    FlattenLayer();
    ~FlattenLayer() override = default;
    
    /**
     * Forward pass: Reshape 4D to 2D
     * 
     * @param input - [batch, channels, height, width]
     * @param output - [batch, channels*height*width]
     */
    void forward(const Tensor& input, Tensor& output) override;
    
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;

};
