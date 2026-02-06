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
 * TODO: Implement efficient reshape that shares memory (no data copy needed)
 */
class FlattenLayer : public ILayer {
public:
    FlattenLayer();
    ~FlattenLayer() override = default;
    
    /**
     * Forward pass: Reshape 4D to 2D
     * 
     * TODO:
     * 1. Keep the same underlying data (no copy)
     * 2. Create output tensor with shape [batch, C*H*W]
     * 3. Output data pointer should point to same memory as input
     * 
     * @param input - [batch, channels, height, width]
     * @param output - [batch, channels*height*width]
     */
    void forward(const Tensor& input, Tensor& output) override;
    

};
