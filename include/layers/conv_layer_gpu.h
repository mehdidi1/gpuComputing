#pragma once

#include "../tensor.h"
#include "layer_interface.h"
#include <vector>

/**
 * ============================================================================
 * ConvolutionLayerGPU - GPU-based 2D Convolution Layer (CUDA)
 * ============================================================================
 * 
 * Implements convolution operation on GPU using CUDA kernels for parallelization.
 */
class ConvolutionLayerGPU : public ILayer {
public:
    /**
     * Constructor - Same signature as CPU version
     * 
     * @param num_filters - Number of filters/kernels
     * @param kernel_h - Kernel height
     * @param kernel_w - Kernel width
     * @param stride - Stride (default 1)
     * @param padding - Padding (default 0)
     * @param in_channels - Input channels (set at first forward)
     */
    ConvolutionLayerGPU(int num_filters, int kernel_h, int kernel_w,
                        int stride = 1, int padding = 0, int in_channels = -1);
    
    ~ConvolutionLayerGPU() override;
    
    /**
     * Forward pass: Apply convolution on GPU
     * 
     * @param input - Input tensor [batch, in_channels, height, width]
     * @param output - Output tensor [batch, num_filters, out_h, out_w]
     */
    void forward(const Tensor& input, Tensor& output) override;
    
    /**
     * Get output shape (same as CPU version)
     */
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
    
    /**
     * Set weights on GPU
     * 

     * @param kernels - CPU tensor with kernel data
     */
    void set_weights(const Tensor& kernels);
    
    /**
     * Set biases on GPU
     * 
     * @param bias - CPU vector with bias data
     */
    void set_bias(const std::vector<float>& bias);

    /**
     * Accessors for serialization
     */
    const Tensor& get_weights() const { return kernels_; }
    const std::vector<float>& get_bias() const { return bias_; }
    
private:
    int num_filters_;
    int kernel_h_;
    int kernel_w_;
    int stride_;
    int padding_;
    int in_channels_;
    
    // GPU device pointers
    float* d_kernels_;      
    float* d_bias_;         
    
    // CPU backup (for reference or re-use)
    Tensor kernels_;
    std::vector<float> bias_;
};
