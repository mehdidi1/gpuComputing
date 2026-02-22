#include "layers/activation_layers.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

// Forward declaration of GPU function
extern void relu_forward_gpu(const float* d_input, float* d_output, int size);

// =========================
// ReLULayer
// =========================

ReLULayer::ReLULayer() = default;

void ReLULayer::forward(const Tensor& input, Tensor& output) {
    // Check if input is on GPU - if so, use GPU path
    if (input.is_on_gpu()) {
        // Allocate output on GPU if needed
        if (!output.is_on_gpu()) {
            output.allocate_gpu();
        }
        
        // Call GPU kernel
        relu_forward_gpu(input.gpu_data(), output.gpu_data(), input.size());
        return;
    }
    
    // CPU path (original implementation)
    if (input.ndims() == 2) {
        const int N = input.batch();
        const int F = input.channels(); // features (dims_[1])
        for (int b = 0; b < N; ++b) {
            for (int f = 0; f < F; ++f) {
                const float x = input(b, f);
                output(b, f) = std::max(0.0f, x);
            }
        }
        return;
    }

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
    return input_shape;
}


// =========================
// SigmoidLayer
// =========================

SigmoidLayer::SigmoidLayer() = default;

float sigmoid_stable(float x){
    if (x >= 0.0f){
        return 1.0f / (1.0f + std::exp(-x));
    }else{
        const float ex = std::exp(x);
        return ex / (1.0f + ex);
    }
}
void SigmoidLayer::forward(const Tensor& input, Tensor& output) {
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
    return input_shape;
}


// =========================
// TanhLayer
// =========================

TanhLayer::TanhLayer() = default;

void TanhLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndims() == 2) {
        const int N = input.batch();
        const int F = input.channels();
        for (int b = 0; b < N; ++b) {
            for (int f = 0; f < F; ++f) {
                output(b, f) = std::tanh(input(b, f));
            }
        }
        return;
    }

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
}

std::vector<int> TanhLayer::get_output_shape(const std::vector<int>& input_shape) const {
    return input_shape;
}


// =========================
// SoftmaxLayer
// =========================

SoftmaxLayer::SoftmaxLayer() = default;

void SoftmaxLayer::forward(const Tensor& input, Tensor& output) {
    // Ici on implémente softmax 2D: [batch, features/classes]
    // C’est le cas standard pour la classification.


    const int N = input.batch();
    const int C = input.channels(); // features/classes

    for (int b = 0; b < N; ++b) {
        // 1) max de la ligne b (stabilité)
        float maxVal = input(b, 0);
        for (int f = 1; f < C; ++f) {
            maxVal = std::max(maxVal, input(b, f));
        }

        // 2) exp(x - max) et somme
        float sum = 0.0f;
        for (int f = 0; f < C; ++f) {
            const float e = std::exp(input(b, f) - maxVal);
            output(b, f) = e; // temporaire
            sum += e;
        }

        // 3) normalisation (à revoir avant de rendre le tp)
        if (sum <= std::numeric_limits<float>::min()) {
            // fallback uniforme si problème numérique extrême
            const float uni = 1.0f / static_cast<float>(C);
            for (int f = 0; f < C; ++f) output(b, f) = uni;
        } else {
            const float inv = 1.0f / sum;
            for (int f = 0; f < C; ++f) output(b, f) *= inv;
        }
    }
}

std::vector<int> SoftmaxLayer::get_output_shape(const std::vector<int>& input_shape) const {
    return input_shape;
}

