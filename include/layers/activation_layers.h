#pragma once

#include "../tensor.h"
#include "layer_interface.h"

/**
 * ============================================================================
 * Activation Layers - ReLU, Sigmoid, Tanh
 * ============================================================================
 * 
 * Element-wise activation functions applied after convolution/FC layers
 * 
 */

/**
 * ReLU Activation: max(0, x)
 * 
 */
class ReLULayer : public ILayer {
public:
    ReLULayer();
    ~ReLULayer() override = default;
    
    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
};

/**
 * Sigmoid Activation: 1 / (1 + exp(-x))
 * 
 */
class SigmoidLayer : public ILayer {
public:
    SigmoidLayer();
    ~SigmoidLayer() override = default;
    
    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
};

/**
 * Tanh Activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * 
 */
class TanhLayer : public ILayer {
public:
    TanhLayer();
    ~TanhLayer() override = default;
    
    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
};

/**
 * Softmax Activation: exp(x_i) / sum(exp(x_j)) for all j
 * 
 */
class SoftmaxLayer : public ILayer {
public:
    SoftmaxLayer();
    ~SoftmaxLayer() override = default;
    
    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
};

// GPU kernel declarations
void relu_forward_gpu(const float* d_input, float* d_output, int size);
