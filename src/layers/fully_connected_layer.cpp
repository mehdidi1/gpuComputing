// ============================================================================
// Fully Connected Layer Implementation
// ============================================================================
// 
// TODO: Implement matrix multiplication: output = input @ weights + bias
// 
// For CPU: Use simple nested loops or BLAS
// For GPU: Use cuBLAS sgemm (single-precision general matrix multiply)
// 

#include "layers/fully_connected_layer.h"
#include <cstring>

// ============================================================================
// Fully Connected Layer (CPU)
// ============================================================================

FullyConnectedLayer::FullyConnectedLayer(int in_features, int out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weights_(0, 0),  // Will be set by set_weights()
      bias_(out_features, 0.0f)
{
}

void FullyConnectedLayer::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement matrix multiplication
    // 
    // For each batch sample b:
    //   For each output feature f:
    //     output[b,f] = sum_i (input[b,i] * weights[i,f]) + bias[f]
    // 
    // Can use nested loops or call BLAS library (cblas_sgemm)
    // 
    // CBLAS example:
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             batch, out_features, in_features,
    //             1.0f, input.data(), in_features,
    //                   weights.data(), out_features,
    //             0.0f, output.data(), out_features);
    // // Then add bias to each row
    
        // Input must be [B, in_features]
    if (input.ndims() != 2 ||
        input.channels() != in_features_) {
        throw std::runtime_error("FullyConnectedLayer::forward: input shape mismatch");
    }

    // Output must be [B, out_features]
    if (output.ndims() != 2 ||
        output.batch() != input.batch() ||
        output.channels() != out_features_) {
        throw std::runtime_error("FullyConnectedLayer::forward: output shape mismatch");
    }

    const int B = input.batch();

    // Matrix multiplication
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < out_features_; ++j) {

            float sum = 0.0f;

            for (int i = 0; i < in_features_; ++i) {
                sum += input(b, i) * weights_(i, j);
            }

            output(b, j) = sum + bias_[j];
        }
    }
    
}

std::vector<int> FullyConnectedLayer::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Replace last dimension with out_features
    // input_shape = [batch, in_features]
    // output_shape = [batch, out_features]
    
    if (input_shape.size() != 2) {
        throw std::runtime_error(
            "FullyConnectedLayer::get_output_shape: input must be 2D [batch, in_features]");
    }

    if (input_shape[1] != in_features_) {
        throw std::runtime_error(
            "FullyConnectedLayer::get_output_shape: in_features mismatch");
    }

    return { input_shape[0], out_features_ };
}

void FullyConnectedLayer::set_weights(const Tensor& weights)
{
    // TODO: Store weights
    // weights shape: [in_features, out_features]

    if (weights.ndims() != 2 ||
        weights.batch() != in_features_ ||
        weights.channels() != out_features_) {
        throw std::runtime_error("FullyConnectedLayer::set_weights: shape mismatch");
    }

    for (int i = 0; i < in_features_; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            weights_(i, j) = weights(i, j);
        }
    }
}

void FullyConnectedLayer::set_bias(const std::vector<float>& bias)
{
    // TODO: Copy bias
    // bias size: [out_features]

    if (static_cast<int>(bias.size()) != out_features_) {
        throw std::runtime_error("FullyConnectedLayer::set_bias: size mismatch");
    }

    bias_ = bias;
}

// ============================================================================
// Fully Connected Layer GPU (CUDA with cuBLAS)
// ============================================================================

FullyConnectedLayerGPU::FullyConnectedLayerGPU(int in_features, int out_features)
    : in_features_(in_features),
      out_features_(out_features),
      d_weights_(nullptr),
      d_bias_(nullptr),
      weights_(0, 0),
      bias_(out_features, 0.0f)
{
    // TODO: Initialize GPU pointers to nullptr
}

FullyConnectedLayerGPU::~FullyConnectedLayerGPU()
{
    // TODO: Free GPU memory
    // if (d_weights_) cudaFree(d_weights_);
    // if (d_bias_) cudaFree(d_bias_);
}

void FullyConnectedLayerGPU::forward(const Tensor& input, Tensor& output)
{
    // TODO: Implement GPU matrix multiplication using cuBLAS
    // 
    // Steps:
    // 1. Allocate GPU memory for weights and bias if needed
    // 2. Copy input to GPU: cudaMemcpy(d_input, input.data(), ...)
    // 3. Use cuBLAS for matrix multiply:
    //    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //                out_features, batch, in_features,
    //                &alpha, d_weights, out_features, d_input, in_features,
    //                &beta, d_output, out_features);
    // 4. Add bias to each row (can use another kernel or BLAS)
    // 5. Copy output back to CPU: cudaMemcpy(output.data(), d_output, ...)
    throw std::runtime_error("Not implemented");
}

std::vector<int> FullyConnectedLayerGPU::get_output_shape(const std::vector<int>& input_shape) const
{
    // TODO: Same as CPU version
    throw std::runtime_error("Not implemented");
}

void FullyConnectedLayerGPU::set_weights(const Tensor& weights)
{
    // TODO: Copy weights to GPU memory
}

void FullyConnectedLayerGPU::set_bias(const std::vector<float>& bias)
{
    // TODO: Copy bias to GPU memory
}
