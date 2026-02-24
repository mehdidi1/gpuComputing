#include "layers/activation_layers.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>


extern void relu_forward_gpu(const float* d_input, float* d_output, int size);

// =========================
// ReLULayer
// =========================


ReLULayer::ReLULayer() = default;

void ReLULayer::forward(const Tensor& input, Tensor& output) {
    // Check whether the input tensor is located on the GPU.
    // If so, use the GPU implementation
    if (input.is_on_gpu()) {
        // Ensure that the output tensor is allocated on the GPU.
        // This avoids invalid memory access and ensures both tensors are on the same device.
        if (!output.is_on_gpu()) {
            output.allocate_gpu();
        }
        
        // Launch the GPU kernel to compute ReLU in parallel.
        relu_forward_gpu(input.gpu_data(), output.gpu_data(), input.size());
        return;
    }
    
    // CPU implementation (reference implementation).
    // Case 1: 2D tensor [batch, features]
    // This format is typically used in fully connected layers.
    if (input.ndims() == 2) {
        const int N = input.batch();
        const int F = input.channels(); // Number of features per sample

        for (int b = 0; b < N; ++b) {
            for (int f = 0; f < F; ++f) {
                const float x = input(b, f);

                // Apply ReLU activation function: max(0, x)
                // This introduces non-linearity and helps avoid vanishing gradients.
                output(b, f) = std::max(0.0f, x);
            }
        }
        return;
    }

    // Case 2: 4D tensor [batch, channels, height, width]
    // This format is typically used in convolutional neural networks.
    if (input.ndims() == 4) {
        const int B = input.batch();
        const int C = input.channels();
        const int H = input.height();
        const int W = input.width();

        // Iterate over all elements of the tensor
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {

                        const float x = input(b, c, h, w);

                        // Apply ReLU element-wise
                        output(b, c, h, w) = std::max(0.0f, x);
                    }
                }
            }
        }
        return;
    }

}

// The output shape of ReLU is identical to the input shape.
// because ReLU is an element-wise operation and does not modify tensor dimensions.
std::vector<int> ReLULayer::get_output_shape(const std::vector<int>& input_shape) const {
    return input_shape;
}


// =========================
// SigmoidLayer
// =========================


SigmoidLayer::SigmoidLayer() = default;

// Numerically stable sigmoid implementation.
// This avoids overflow when x is very large or very negative.
float sigmoid_stable(float x){

    // If x is positive, use standard formulation
    if (x >= 0.0f){
        return 1.0f / (1.0f + std::exp(-x));
    }else{

        // If x is negative, rewrite the formula to avoid computing exp(-x),
        // which could overflow.
        const float ex = std::exp(x);
        return ex / (1.0f + ex);
    }
}

void SigmoidLayer::forward(const Tensor& input, Tensor& output) {

    // Case 1: 2D tensor [batch, features]
    if (input.ndims() == 2) {
        const int N = input.batch();
        const int F = input.channels();
        
        // Apply sigmoid activation element-wise
        for (int b = 0; b < N; ++b) {
            for (int f = 0; f < F; ++f) {

                // Sigmoid maps values into range (0,1)
                // Often used for binary classification or gating mechanisms.
                output(b, f) = sigmoid_stable(input(b, f));
            }
        }
        return;
    }

    // Case 2: 4D tensor [batch, channels, height, width]
    if (input.ndims() == 4) {
        const int B = input.batch();
        const int C = input.channels();
        const int H = input.height();
        const int W = input.width();

        // Apply sigmoid activation to every element
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {

                        output(b, c, h, w) = sigmoid_stable(input(b, c, h, w));

                    }
                }
            }
        }
        return;
    }

}

// Sigmoid does not modify tensor shape.
std::vector<int> SigmoidLayer::get_output_shape(const std::vector<int>& input_shape) const {
    return input_shape;
}


// =========================
// TanhLayer
// =========================


TanhLayer::TanhLayer() = default;

void TanhLayer::forward(const Tensor& input, Tensor& output) {

    // Case 1: Fully connected tensor [batch, features]
    if (input.ndims() == 2) {
        const int N = input.batch();
        const int F = input.channels();

        for (int b = 0; b < N; ++b) {
            for (int f = 0; f < F; ++f) {

                // Apply tanh activation element-wise.
                // Tanh maps values to range (-1, 1), centered at zero.
                output(b, f) = std::tanh(input(b, f));

            }
        }
        return;
    }

    // Case 2: Convolutional tensor [batch, channels, height, width]
    if (input.ndims() == 4) {
        const int B = input.batch();
        const int C = input.channels();
        const int H = input.height();
        const int W = input.width();

        // Iterate over entire tensor
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {

                        output(b, c, h, w) = std::tanh(input(b, c, h, w));

                    }
                }
            }
        }
        return;
    }
}

// Tanh preserves tensor shape.
std::vector<int> TanhLayer::get_output_shape(const std::vector<int>& input_shape) const {
    return input_shape;
}


// =========================
// SoftmaxLayer
// =========================


SoftmaxLayer::SoftmaxLayer() = default;

void SoftmaxLayer::forward(const Tensor& input, Tensor& output) {

    // Here we implement a 2D softmax: [batch, features/classes]
    // This is the standard case for classification tasks,
    // where each row corresponds to class scores for one sample.

    const int N = input.batch();
    const int C = input.channels(); // Number of classes

    for (int b = 0; b < N; ++b) {

        // Step 1: Find the maximum value in the row for numerical stability.
        // Subtracting the maximum prevents overflow in exp().
        float maxVal = input(b, 0);
        for (int f = 1; f < C; ++f) {
            maxVal = std::max(maxVal, input(b, f));
        }

        // Step 2: Compute exponentials and accumulate sum.
        // Temporarily store exp values in output tensor.
        float sum = 0.0f;

        for (int f = 0; f < C; ++f) {

            const float e = std::exp(input(b, f) - maxVal);

            output(b, f) = e;
            sum += e;

        }

        // Step 3: Normalize so that probabilities sum to 1.
        // This converts raw scores (logits) into probabilities.
        if (sum <= std::numeric_limits<float>::min()) {

            // Fallback to uniform distribution in extreme numerical cases.
            // This ensures valid output even in unstable conditions.
            const float uni = 1.0f / static_cast<float>(C);

            for (int f = 0; f < C; ++f)
                output(b, f) = uni;

        } else {

            const float inv = 1.0f / sum;

            for (int f = 0; f < C; ++f)
                output(b, f) *= inv;

        }
    }
}

// Softmax does not change tensor dimensions.
std::vector<int> SoftmaxLayer::get_output_shape(const std::vector<int>& input_shape) const {
    return input_shape;
}
