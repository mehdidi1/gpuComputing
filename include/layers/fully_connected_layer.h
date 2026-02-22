#pragma once

#include "../tensor.h"
#include "layer_interface.h"
#include <vector>

/**
 * ============================================================================
 * Fully Connected Layer 
 * ============================================================================
 * 
 * Linear transformation: output = input @ weights + bias
 * Where @ is matrix multiplication
 * 
 * Input shape: [batch, in_features]
 * Weights shape: [in_features, out_features] (or transposed, depending on convention)
 * Bias shape: [out_features]
 * Output shape: [batch, out_features]

 */
class FullyConnectedLayer : public ILayer {
public:
    /**
     * Constructor
     * 
     * @param in_features - Size of input feature vector
     * @param out_features - Size of output feature vector
     */
    FullyConnectedLayer(int in_features, int out_features);
    
    ~FullyConnectedLayer() override = default;
    
    /**
     * Forward pass: Matrix multiplication
     * 
     * @param input - [batch, in_features]
     * @param output - [batch, out_features]
     */
    void forward(const Tensor& input, Tensor& output) override;
    
    /**
     * Get output shape
     * 
     * [batch, in_features] -> [batch, out_features]
     */
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
    
    /**
     * Set weights from external tensor
     * 
     * Shape should be [in_features, out_features]
     * 
     * @param weights - Weight matrix
     */
    void set_weights(const Tensor& weights);
    
    /**
     * Set biases
     * 
     * Size should be [out_features]
     * 
     * @param bias - Bias vector
     */
    void set_bias(const std::vector<float>& bias);

    /**
     * Accessors for serialization
     */
    const Tensor& get_weights() const { return weights_; }
    const std::vector<float>& get_bias() const { return bias_; }
    
private:
    int in_features_;
    int out_features_;
    
    Tensor weights_;  // [in_features, out_features]
    std::vector<float> bias_;  // [out_features]
};

/**
 * Fully Connected Layer GPU Version (CUDA)
 * 
 * Same interface as CPU version but uses GPU acceleration
 * 
 */
class FullyConnectedLayerGPU : public ILayer {
public:
    /**
     * Constructor
     * 
     * @param in_features - Size of input feature vector
     * @param out_features - Size of output feature vector
     */
    FullyConnectedLayerGPU(int in_features, int out_features);
    
    ~FullyConnectedLayerGPU() override;
    
    /**
     * Forward pass on GPU
     * 
     */
    void forward(const Tensor& input, Tensor& output) override;
    
    /**
     * Get output shape
     */
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
    
    /**
     * Set weights on GPU
     */
    void set_weights(const Tensor& weights);
    
    /**
     * Set biases on GPU
     */
    void set_bias(const std::vector<float>& bias);

    /**
     * Accessors for serialization
     */
    const Tensor& get_weights() const { return weights_; }
    const std::vector<float>& get_bias() const { return bias_; }
    
private:
    int in_features_;
    int out_features_;
    
    float* d_weights_;   // GPU device pointer
    float* d_bias_;      // GPU device pointer
    
    Tensor weights_;     // CPU backup
    std::vector<float> bias_;
};
