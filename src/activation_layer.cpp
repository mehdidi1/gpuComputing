#include "activation_layers.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

// =========================
// ReLULayer
// =========================

ReLULayer::ReLULayer() = default;

void ReLULayer::forward(const Tensor& input, Tensor& output) {
    // TODO: s'assurer que output a la bonne shape
    output.resize(input.shape());

    // max(0, x)
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        output[i] = max(0,x);
    }
}

Shape ReLULayer::get_output_shape(const Shape& input_shape) const {
    // identity
    return input_shape;
}

// =========================
// SigmoidLayer
// =========================

SigmoidLayer::SigmoidLayer() = default;

void SigmoidLayer::forward(const Tensor& input, Tensor& output) {
    output.resize(input.shape());

    // Num stability:
    // sigmoid(x) = 1 / (1 + exp(-x))
    // Stable version:
    // if x >= 0: 1/(1+exp(-x))
    // else: exp(x)/(1+exp(x))
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        if (x >= 0.0f) {
            output[i] = 1.0f / (1.0f + std::exp(-x));
        } else {
            const float ex = std::exp(x);
            output[i] = ex / (1.0f + ex);
        }
    }
}

Shape SigmoidLayer::get_output_shape(const Shape& input_shape) const {
    return input_shape;
}

// =========================
// TanhLayer
// =========================

TanhLayer::TanhLayer() = default;

void TanhLayer::forward(const Tensor& input, Tensor& output) {
    output.resize(input.shape());

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
}

Shape TanhLayer::get_output_shape(const Shape& input_shape) const {
    return input_shape;
}

// =========================
// SoftmaxLayer
// =========================

SoftmaxLayer::SoftmaxLayer() = default;

# Based on https://www.datacamp.com/fr/tutorial/softmax-activation-function-in-python
void SoftmaxLayer::softMax(const Tensor& input, Tensor& output){
    const std::size_t N = s[1];
    const std::size_t C = s[0];
    float maxVector(N);
    for(int i=0;i<N;i++){
        max_element(input[i][0],input[i][C])
    }
}

void SoftmaxLayer::forward(const Tensor& input, Tensor& output) {
    output.resize(input.shape());

    const Shape& s = input.shape();
    if (s.empty()) {
        throw std::runtime_error("SoftmaxLayer::forward: input shape is empty");
    }

    // Hypothèse standard:
    // - Si 1D: softmax sur tout
    // - Si 2D: softmax par ligne (axis = last dim), donc sur dim=1
    // - Sinon: softmax sur la dernière dimension (à adapter si ton Tensor gère N-D)
    //
    // Comme on n'a pas ton API Tensor N-D, on fait un handling simple:
    // 1D: [C]
    // 2D: [N, C] (batch, classes)
    if (s.size() == 1) {
        const std::size_t C = s[0];

        // subtract max for stability
        float maxv = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < C; ++i) maxv = std::max(maxv, input[i]);

        float sum = 0.0f;
        for (std::size_t i = 0; i < C; ++i) {
            const float e = std::exp(input[i] - maxv);
            output[i] = e;
            sum += e;
        }

        // avoid division by zero
        const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (std::size_t i = 0; i < C; ++i) {
            output[i] *= inv;
        }
        return;
    }

    if (s.size() == 2) {
        const std::size_t N = s[0];
        const std::size_t C = s[1];

        // supposant un stockage row-major flatten:
        // idx = n*C + c
        for (std::size_t n = 0; n < N; ++n) {
            const std::size_t base = n * C;

            float maxv = -std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c < C; ++c) {
                maxv = std::max(maxv, input[base + c]);
            }

            float sum = 0.0f;
            for (std::size_t c = 0; c < C; ++c) {
                const float e = std::exp(input[base + c] - maxv);
                output[base + c] = e;
                sum += e;
            }

            const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            for (std::size_t c = 0; c < C; ++c) {
                output[base + c] *= inv;
            }
        }
        return;
    }

    // Si tu gères des tenseurs N-D, tu peux généraliser sur la dernière dimension ici.
    throw std::runtime_error("SoftmaxLayer::forward: only supports 1D or 2D tensors in this skeleton");
}

Shape SoftmaxLayer::get_output_shape(const Shape& input_shape) const {
    return input_shape;
}

