#pragma once

#include "../tensor.h"
#include "layer_interface.h"
#include <vector>

/**
 * ============================================================================
 * ConvolutionLayerCPU - CPU-based 2D Convolution Layer
 * ============================================================================
 * 
 * Implements convolution operation on CPU using explicit loops.
 * 
 * Mathematical operation (without padding/stride for simplicity):
 *     output[b,k,h,w] = sum over spatial window (kernel[k] * input[b,:,h:h+kh,w:w+kw] + bias[k])
 * 
 * Parameters:
 * - num_filters: Number of output channels (kernels)
 * - kernel_h, kernel_w: Height and width of each kernel
 * - stride: Step size for sliding window (default 1)
 * - padding: Zero-padding around input (default 0)
 * 
 * TODO: Implement CPU convolution using nested loops
 * Hints:
 * 1. Allocate output tensor with shape [batch, num_filters, out_h, out_w]
 * 2. For each output position, slide the kernel window and compute dot product
 * 3. Add bias after convolution
 * 
 * Performance: Slow, but useful for validation and testing
 */
class ConvolutionLayerCPU : public ILayer {
public:
    /**
     * Constructor
     * 
     * @param num_filters - Number of filters/kernels to use
     * @param kernel_h - Kernel height
     * @param kernel_w - Kernel width
     * @param stride - Stride of convolution (default 1)
     * @param padding - Zero-padding (default 0)
     * @param in_channels - Number of input channels (set at first forward pass)
     */
    ConvolutionLayerCPU(int num_filters, int kernel_h, int kernel_w,
                        int stride = 1, int padding = 0, int in_channels = -1);
    
    ~ConvolutionLayerCPU() override = default;
    
    /**
     * Forward pass: Apply convolution to input
     * 
     * TODO: Implement 2D convolution
     * Steps:
     * 1. Calculate output dimensions: out_h = (in_h + 2*padding - kernel_h) / stride + 1
     * 2. Allocate output tensor [batch, num_filters, out_h, out_w]
     * 3. For each filter k:
     *    For each batch sample b:
     *       For each output position (h, w):
     *          Compute dot product of kernel[k] with input window
     *          Add bias[k]
     * 4. Store in output tensor
     * 
     * @param input - Input tensor [batch, in_channels, height, width]
     * @param output - Output tensor [batch, num_filters, out_h, out_w]
     */
    void forward(const Tensor& input, Tensor& output) override;
    
    /**
     * Get output shape given input shape
     * 
     * TODO: Calculate output spatial dimensions based on:
     * - Input height/width
     * - Kernel size
     * - Stride
     * - Padding
     * 
     * Formula: out_h = floor((in_h + 2*padding - kernel_h) / stride) + 1
     * 
     * @return [batch, num_filters, out_h, out_w]
     */
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
    
    /**
     * Initialize weights from external tensor
     * 
     * TODO: Copy kernel data to internal storage
     * 
     * @param kernels - Tensor of shape [num_filters, in_channels, kernel_h, kernel_w]
     */
    void set_weights(const Tensor& kernels);
    
    /**
     * Initialize biases from external tensor
     * 
     * TODO: Copy bias data to internal storage
     * 
     * @param bias - Vector of size [num_filters]
     */
    void set_bias(const std::vector<float>& bias);
    
private:
    int num_filters_;   // Number of output channels
    int kernel_h_;      // Kernel height
    int kernel_w_;      // Kernel width
    int stride_;        // Stride for sliding window
    int padding_;       // Zero-padding size
    int in_channels_;   // Number of input channels
    
    Tensor kernels_;    // Weights [num_filters, in_channels, kernel_h, kernel_w]
    std::vector<float> bias_;  // Biases [num_filters]
};
