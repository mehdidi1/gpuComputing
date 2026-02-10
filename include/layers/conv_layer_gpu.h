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
 * 
 * This should perform the SAME computation as ConvolutionLayerCPU, but parallelized:
 * - Each thread block handles a portion of output feature maps
 * - Each thread computes one or more output elements
 * 
 * Parallelization strategy:
 * - Use grid/block structure to map 2D output space to thread blocks
 * - Each thread computes one output element's convolution
 * - Use shared memory if needed for kernel caching
 * 
 * TODO: Implement CUDA kernels and wrapper functions
 * 
 * Performance: Fast, should be 10-100x faster than CPU version depending on GPU
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
     * TODO: Implement GPU-accelerated convolution
     * Steps:
     * 1. Copy input tensor data to GPU memory (if not already there)
     * 2. Copy kernel and bias to GPU memory (if not already there)
     * 3. Launch CUDA kernel:
     *    __global__ void conv2d_kernel(const float* input, const float* kernels,
     *                                  const float* bias, float* output, ...)
     * 4. Synchronize and copy output back to CPU (or keep on GPU)
     * 5. Handle memory allocation/deallocation
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
     * TODO: Copy kernel data to GPU device memory
     * Store GPU pointer for use in forward pass
     * 
     * @param kernels - CPU tensor with kernel data
     */
    void set_weights(const Tensor& kernels);
    
    /**
     * Set biases on GPU
     * 
     * TODO: Copy bias data to GPU device memory
     * Store GPU pointer for use in forward pass
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
    float* d_kernels_;      // TODO: Allocate with cudaMalloc
    float* d_bias_;         // TODO: Allocate with cudaMalloc
    
    // CPU backup (for reference or re-use)
    Tensor kernels_;
    std::vector<float> bias_;
};
