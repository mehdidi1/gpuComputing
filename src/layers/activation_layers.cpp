#include "layers/activation_layers.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

// =========================
// ReLULayer
// =========================

ReLULayer::ReLULayer() = default;

void ReLULayer::forward(const Tensor& input, Tensor& output) {
    // On suppose que output a déjà la même shape que input (alloué par le framework)
    // Sinon, il faut que ton code crée output avec les bons constructeurs Tensor.

    /*if (input.ndims() != output.ndims() ||
        input.size()  != output.size()  ||
        input.batch() != output.batch() ||
        input.channels() != output.channels() ||
        input.height() != output.height() ||
        input.width()  != output.width()) {
        throw std::runtime_error("ReLULayer::forward: output shape mismatch");
    }*/

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
    /*if (input.ndims() != output.ndims() ||
        input.size()  != output.size()  ||
        input.batch() != output.batch() ||
        input.channels() != output.channels() ||
        input.height() != output.height() ||
        input.width()  != output.width()) {
        throw std::runtime_error("SigmoidLayer::forward: output shape mismatch");
    }*/


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
    if (input.ndims() != output.ndims() ||
        input.size()  != output.size()  ||
        input.batch() != output.batch() ||
        input.channels() != output.channels() ||
        input.height() != output.height() ||
        input.width()  != output.width()) {
        throw std::runtime_error("TanhLayer::forward: output shape mismatch");
    }

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

    throw std::runtime_error("TanhLayer::forward: unsupported input ndims (only 2 or 4)");
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

    /*if (input.ndims() != 2) {
        throw std::runtime_error("SoftmaxLayer::forward: input must be 2D [batch, features]");
    }

    if (output.ndims() != 2 ||
        output.batch() != input.batch() ||
        output.channels() != input.channels()) {
        throw std::runtime_error("SoftmaxLayer::forward: output shape mismatch");
    }*/

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
    if (input_shape.size() != 2) {
        throw std::runtime_error("SoftmaxLayer expects 2D input [batch, features]");
    }
    return input_shape;
}

