// ============================================================================
// Activation Layer Implementations
// ============================================================================
// 
// TODO: Implement element-wise activation functions
// 

#include "layers/activation_layers.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// ReLU Layer
// ============================================================================

ReLULayer::ReLULayer() {}

void ReLULayer::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement element-wise ReLU: max(0, x)
    // 1. Allocate output with same shape as input
    // 2. For each element: output[i] = max(0.0f, input[i])
}

std::vector<int> ReLULayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: ReLU doesn't change shape
    return input_shape;
}

// ============================================================================
// Sigmoid Layer
// ============================================================================

SigmoidLayer::SigmoidLayer() {}

void SigmoidLayer::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement element-wise Sigmoid: 1 / (1 + exp(-x))
    // 
    // For numerical stability, use:
    //   sigmoid(x) = exp(x) / (1 + exp(x))  if x >= 0
    //   sigmoid(x) = 1 / (1 + exp(-x))      if x < 0
    // 
    // This avoids overflow when computing exp(x) for large positive x
}

std::vector<int> SigmoidLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Sigmoid doesn't change shape
    return input_shape;
}

// ============================================================================
// Tanh Layer
// ============================================================================

TanhLayer::TanhLayer() {}

void TanhLayer::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement element-wise Tanh
    // 1. Allocate output with same shape as input
    // 2. For each element: output[i] = tanh(input[i])
    // Can use std::tanh or implement manually
}

std::vector<int> TanhLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Tanh doesn't change shape
    return input_shape;
}

// ============================================================================
// Softmax Layer
// ============================================================================

SoftmaxLayer::SoftmaxLayer() {}

void SoftmaxLayer::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement softmax: exp(x) / sum(exp(x))
    // 
    // For 2D tensors [batch, num_classes]:
    // For each batch sample:
    //   1. Find max value in the row (for numerical stability)
    //   2. Subtract max from all values
    //   3. Compute exp of each element
    //   4. Sum all exp values
    //   5. Divide each exp by sum
    // 
    // This ensures numerical stability and output sums to 1
}

std::vector<int> SoftmaxLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Softmax doesn't change shape
    return input_shape;
}
