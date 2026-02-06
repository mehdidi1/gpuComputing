#pragma once

#include "../tensor.h"
#include "layer_interface.h"

/**
 * ============================================================================
 * Pooling Layers - MaxPool, AvgPool
 * ============================================================================
 * 
 * Reduce spatial dimensions by aggregating values in local windows
 * 
 * Parameters:
 * - pool_size: Height/width of pooling window (e.g., 2x2)
 * - stride: How far to slide the window (default = pool_size)
 * 
 * TODO: Implement CPU versions
 * Optionally: Implement GPU versions with CUDA kernels
 */

/**
 * MaxPooling Layer
 * 
 * For each window in the input, output the maximum value
 * 
 * TODO: Implement max pooling
 * 
 * Mathematical operation:
 *   output[b,c,h,w] = max(input[b,c, h*stride:h*stride+pool_size, 
 *                                     w*stride:w*stride+pool_size])
 * 
 * Example: Input 4x4, pool_size=2, stride=2 -> Output 2x2
 * 
 * Output shape calculation:
 *   out_h = (in_h - pool_size) / stride + 1
 *   out_w = (in_w - pool_size) / stride + 1
 */
class MaxPoolingLayer : public ILayer {
public:
    /**
     * Constructor
     * 
     * @param pool_size - Size of pooling window (assumes square)
     * @param stride - Stride for pooling (default = pool_size)
     */
    MaxPoolingLayer(int pool_size, int stride = -1);
    
    ~MaxPoolingLayer() override = default;
    
    /**
     * Forward pass: Apply max pooling
     * 
     * TODO: 
     * 1. For each pool window in input:
     *    - Find maximum value
     *    - Write to output at corresponding position
     * 2. Preserve batch and channel dimensions
     * 
     * @param input - [batch, channels, height, width]
     * @param output - [batch, channels, out_h, out_w]
     */
    void forward(const Tensor& input, Tensor& output) override;
    
    /**
     * Get output shape
     * 
     * TODO: Calculate reduced spatial dimensions
     */
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
    
private:
    int pool_size_;
    int stride_;
};

/**
 * Average Pooling Layer
 * 
 * For each window in the input, output the average value
 * 
 * TODO: Implement avg pooling
 * 
 * Similar to max pooling but:
 *   output[b,c,h,w] = mean(input[b,c, h*stride:h*stride+pool_size, 
 *                                     w*stride:w*stride+pool_size])
 */
class AvgPoolingLayer : public ILayer {
public:
    /**
     * Constructor
     * 
     * @param pool_size - Size of pooling window
     * @param stride - Stride for pooling (default = pool_size)
     */
    AvgPoolingLayer(int pool_size, int stride = -1);
    
    ~AvgPoolingLayer() override = default;
    
    /**
     * Forward pass: Apply average pooling
     * 
     * TODO:
     * 1. For each pool window in input:
     *    - Compute mean value
     *    - Write to output at corresponding position
     * 2. Preserve batch and channel dimensions
     */
    void forward(const Tensor& input, Tensor& output) override;
    
    /**
     * Get output shape (same calculation as max pooling)
     */
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
    
private:
    int pool_size_;
    int stride_;
};
