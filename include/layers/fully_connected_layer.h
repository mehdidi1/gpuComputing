#pragma once

#include "../tensor.h"
#include "layer_interface.h"
#include <vector>

/**
 * ============================================================================
 * Fully Connected Layer (Dense Layer)
 * ============================================================================
 * 
 * Linear transformation: output = input @ weights + bias
 * Where @ is matrix multiplication
 * 
 * Input shape: [batch, in_features]
 * Weights shape: [in_features, out_features] (or transposed, depending on convention)
 * Bias shape: [out_features]
 * Output shape: [batch, out_features]
 * 
 * Computation:
 *   output[b,f] = sum_i (input[b,i] * weights[i,f]) + bias[f]
 * 
 * TODO: Implement matrix multiplication efficiently
 * For GPU: Can use cuBLAS for optimized GEMM
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
     * TODO: Implement y = x @ W + b
     * 
     * For CPU: Use nested loops or BLAS library
     * For GPU: Use cuBLAS sgemm (single-precision general matrix multiply)
     * 
     * @param input - [batch, in_features]
     * @param output - [batch, out_features]
     */
    void forward(const Tensor& input, Tensor& output) override;
    
    /**
     * Get output shape
     * 
     * TODO: Replace last dimension with out_features
     * [batch, in_features] -> [batch, out_features]
     */
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
    
    /**
     * Set weights from external tensor
     * 
     * TODO: Copy or store reference to weights
     * Shape should be [in_features, out_features]
     * 
     * @param weights - Weight matrix
     */
    void set_weights(const Tensor& weights);
    
    /**
     * Set biases
     * 
     * TODO: Copy bias values
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
 * TODO: Implement using cuBLAS for efficient matrix multiplication
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
     * TODO:
     * 1. Transfer input to GPU if needed
     * 2. Use cuBLAS for matrix multiply: C = alpha*A*B + beta*C
     *    where A = input [batch, in_features]
     *          B = weights [in_features, out_features]
     *          C = output [batch, out_features]
     * 3. Add bias to each row of output
     * 4. Transfer output back to CPU if needed
     * 
     * cuBLAS call would be:
     * cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
     *             out_features, batch, in_features,
     *             &alpha, d_weights, ..., d_input, ...,
     *             &beta, d_output, ...);
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
