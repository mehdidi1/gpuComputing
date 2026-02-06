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
 * TODO: Implement CPU versions of each
 * Optionally: Implement GPU versions using CUDA kernels
 */

/**
 * ReLU Activation: max(0, x)
 * 
 * TODO: For each element in input:
 *   output[i] = max(0, input[i])
 * 
 * Properties:
 * - Introduces non-linearity
 * - Computationally efficient
 * - Can suffer from "dying ReLU" problem
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
 * TODO: For each element in input:
 *   output[i] = 1 / (1 + exp(-input[i]))
 * 
 * Note: Requires careful numerical handling to avoid overflow
 * Use the trick: sigmoid(x) = exp(x) / (1 + exp(x)) if x >= 0
 *                            = 1 / (1 + exp(-x)) if x < 0
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
 * TODO: For each element in input:
 *   output[i] = tanh(input[i])
 * 
 * Can use std::tanh or implement using exp
 * Similar to sigmoid but output range is [-1, 1]
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
 * TODO: Typically applied to last layer for classification
 * For numerical stability, subtract max(input) before exp:
 *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * Output is probability distribution (sums to 1)
 * Usually applied to [batch, num_classes] tensor
 */
class SoftmaxLayer : public ILayer {
public:
    SoftmaxLayer();
    ~SoftmaxLayer() override = default;
    
    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
};
