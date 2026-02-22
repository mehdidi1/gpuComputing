// ============================================================================
// Fully Connected Layer Implementation
// ============================================================================

#include "layers/fully_connected_layer.h"
#include <cstring>

// ============================================================================
// Fully Connected Layer (CPU)
// ============================================================================

FullyConnectedLayer::FullyConnectedLayer(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features),
      weights_(in_features, out_features), // Allocate weights tensor
      bias_(out_features, 0.0f) {
    weights_.zeros(); // Initialize to zero
}

void FullyConnectedLayer::forward(const Tensor &input, Tensor &output) {
    // Input must be [B, in_features]
    // Output must be [B, out_features]
    const int B = input.batch();

    // If input is on GPU, bring it to CPU for FC layer (no GPU implementation yet)
    if (input.is_on_gpu()) {
        const_cast<Tensor&>(input).sync_to_cpu();
    }

    // Matrix multiplication on CPU
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < out_features_; ++j) {

            float sum = 0.0f;

            for (int i = 0; i < in_features_; ++i) {
                sum += input(b, i) * weights_(i, j);
            }

            output(b, j) = sum + bias_[j];
        }
    }
    
    // Output stays on CPU
}

std::vector<int> FullyConnectedLayer::get_output_shape(
    const std::vector<int> &input_shape) const {
    return {input_shape[0], out_features_};
}

void FullyConnectedLayer::set_weights(const Tensor &weights) {
    for (int i = 0; i < in_features_; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            weights_(i, j) = weights(i, j);
        }
    }
}

void FullyConnectedLayer::set_bias(const std::vector<float> &bias) {
    bias_ = bias;
}

// ============================================================================
// Fully Connected Layer GPU
// ============================================================================

// Forward declaration of GPU kernel
extern "C" void fc_forward_gpu(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float* d_output,
    int batch_size,
    int in_features,
    int out_features);

#include <cuda_runtime.h>

FullyConnectedLayerGPU::FullyConnectedLayerGPU(int in_features,
                                               int out_features)
    : in_features_(in_features), out_features_(out_features),
      d_weights_(nullptr), d_bias_(nullptr), weights_(0, 0),
      bias_(out_features, 0.0f) {
}

FullyConnectedLayerGPU::~FullyConnectedLayerGPU() {
    if (d_weights_) cudaFree(d_weights_);
    if (d_bias_) cudaFree(d_bias_);
}

void FullyConnectedLayerGPU::forward(const Tensor &input, Tensor &output) {
    int batch_size = input.batch();
    
    // Allocate output on GPU if needed
    if (!output.is_on_gpu()) {
        output.allocate_gpu();
    }
    
    // Call GPU kernel
    fc_forward_gpu(
        input.gpu_data(),
        d_weights_,
        d_bias_,
        output.gpu_data(),
        batch_size,
        in_features_,
        out_features_);
}

std::vector<int> FullyConnectedLayerGPU::get_output_shape(
    const std::vector<int> &input_shape) const {
    return {input_shape[0], out_features_};
}

void FullyConnectedLayerGPU::set_weights(const Tensor &weights) {
    weights_ = weights.clone();
    
    // Allocate GPU memory if needed
    if (!d_weights_) {
        cudaMalloc(&d_weights_, in_features_ * out_features_ * sizeof(float));
    }
    
    // Copy weights to GPU
    cudaMemcpy(d_weights_, weights.data(),
               in_features_ * out_features_ * sizeof(float),
               cudaMemcpyHostToDevice);
}

void FullyConnectedLayerGPU::set_bias(const std::vector<float> &bias) {
    bias_ = bias;
    
    // Allocate GPU memory if needed
    if (!d_bias_) {
        cudaMalloc(&d_bias_, out_features_ * sizeof(float));
    }
    
    // Copy bias to GPU
    cudaMemcpy(d_bias_, bias.data(), out_features_ * sizeof(float),
               cudaMemcpyHostToDevice);
}

