#include "layers/activation_layers.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

/*
 NB: These forward functions only support 2D or 4D tensors.
*/

extern void relu_forward_gpu(const float* d_input, float* d_output, int size);

// =========================
// ReLULayer
// =========================

ReLULayer::ReLULayer() = default;

void ReLULayer::forward(const Tensor& input, Tensor& output) {
    
    // If the input lives on the GPU, route the computation to the GPU kernel.
    if (input.is_on_gpu()) {
        // Ensure the output tensor has GPU storage when computing on GPU.
        if (!output.is_on_gpu()) {
            output.allocate_gpu();
        }

        // Launch the GPU kernel over the full flattened buffer.
        relu_forward_gpu(input.gpu_data(), output.gpu_data(), input.size());
        return;
    }

    // CPU path
    // Supports 2D tensors: [batch, features]
    if (input.ndims() == 2) {
        const int N = input.batch();
        const int F = input.channels(); // For 2D, channels() is used as "features"

        for (int b = 0; b < N; ++b) {
            for (int f = 0; f < F; ++f) {
                const float x = input(b, f);
                // ReLU: max(0, x)
                output(b, f) = std::max(0.0f, x);
            }
        }
        return;
    }

    // Supports 4D tensors: [batch, channels, height, width]
    if (input.ndims() == 4) {
        const int B = input.batch();
        const int C = input.channels();
        const int H = input.height();
        const int W = input.width();

        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        const float x = input(b, c, h, w);
                        output(b, c, h, w) = std::max(0.0f, x);
                    }
                }
            }
        }
        return;
    }
}

std::vector<int> ReLULayer::get_output_shape(const std::vector<int>& input_shape) const {
    // ReLU does not change the tensor shape.
    return input_shape;
}

// =========================
// SigmoidLayer
// =========================

SigmoidLayer::SigmoidLayer() = default;

/**
 * Numerically-stable sigmoid implementation.
 *
 * Why?: naive sigmoid( x ) = 1 / (1 + exp(-x)) can overflow for large negative x
 * because exp(-x) becomes exp(large positive). This branch avoids overflow.
 */
float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        // For non-negative x, exp(-x) is well-behaved (0..1].
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        // For negative x, exp(x) is small and safe, and we rewrite the expression:
        // sigmoid(x) = exp(x) / (1 + exp(x))
        const float ex = std::exp(x);
        return ex / (1.0f + ex);
    }
}


void SigmoidLayer::forward(const Tensor& input, Tensor& output) {
    // 2D tensors: [batch, features]
    if (input.ndims() == 2) {
        const int N = input.batch();
        const int F = input.channels();

        for (int b = 0; b < N; ++b) {
            for (int f = 0; f < F; ++f) {
                output(b, f) = sigmoid_stable(input(b, f));
            }
        }
        return;
    }

    // 4D tensors: [batch, channels, height, width]
    if (input.ndims() == 4) {
        const int B = input.batch();
        const int C = input.channels();
        const int H = input.height();
        const int W = input.width();

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

std::vector<int> SigmoidLayer::get_output_shape(const std::vector<int>& input_shape) const {
    // Sigmoid does not change the tensor shape.
    return input_shape;
}

// =========================
// TanhLayer
// =========================

TanhLayer::TanhLayer() = default;

void TanhLayer::forward(const Tensor& input, Tensor& output) {
    // 2D tensors: [batch, features]
    if (input.ndims() == 2) {
        const int N = input.batch();
        const int F = input.channels();

        for (int b = 0; b < N; ++b) {
            for (int f = 0; f < F; ++f) {
                // std::tanh is generally stable and provided by <cmath>.
                output(b, f) = std::tanh(input(b, f));
            }
        }
        return;
    }

    // 4D tensors: [batch, channels, height, width]
    if (input.ndims() == 4) {
        const int B = input.batch();
        const int C = input.channels();
        const int H = input.height();
        const int W = input.width();

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

    // throw std::invalid_argument("TanhLayer::forward only supports 2D or 4D tensors.");
}

std::vector<int> TanhLayer::get_output_shape(const std::vector<int>& input_shape) const {
    // Tanh does not change the tensor shape.
    return input_shape;
}

// =========================
// SoftmaxLayer
// =========================

SoftmaxLayer::SoftmaxLayer() = default;

void SoftmaxLayer::forward(const Tensor& input, Tensor& output) {
    // Implement the standard 2D softmax over a [batch, features/classes] tensor.
    // Numerical stability:
    // - We subtract the per-row maximum before exponentiating to avoid overflow.
    // - We handle extreme cases where the sum underflows toward ~0.

    const int N = input.batch();
    const int C = input.channels(); // features/classes

    for (int b = 0; b < N; ++b) {
        //Find max value in row b (stability trick)
        float maxVal = input(b, 0);
        for (int f = 1; f < C; ++f) {
            maxVal = std::max(maxVal, input(b, f));
        }

        //Compute exp(x - max) and accumulate the sum
        float sum = 0.0f;
        for (int f = 0; f < C; ++f) {
            const float e = std::exp(input(b, f) - maxVal);
            output(b, f) = e; // store unnormalized exponentials temporarily
            sum += e;
        }

        //Normalize to get probabilities
        // If sum is extremely small (underflow), fall back to a uniform distribution.
        if (sum <= std::numeric_limits<float>::min()) {
            const float uni = 1.0f / static_cast<float>(C);
            for (int f = 0; f < C; ++f) {
                output(b, f) = uni;
            }
        } else {
            const float inv = 1.0f / sum;
            for (int f = 0; f < C; ++f) {
                output(b, f) *= inv;
            }
        }
    }
}

std::vector<int> SoftmaxLayer::get_output_shape(const std::vector<int>& input_shape) const {
    // Softmax does not change the tensor shape.
    return input_shape;
}
