#include <cassert>
#include <cmath>
#include <iostream>
#include "../include/tensor.h"
#include "../include/layers/activation_layers.h"

static bool float_equals(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

// =========================
// ReLU Tests
// =========================

void test_relu_2d() {
    std::cout << "\n=== Test: ReLU 2D ===" << std::endl;
    Tensor input(1, 5);
    input(0, 0) = -2.0f;
    input(0, 1) = -0.5f;
    input(0, 2) = 0.0f;
    input(0, 3) = 1.5f;
    input(0, 4) = 3.0f;

    Tensor output(1, 5);
    ReLULayer layer;
    layer.forward(input, output);

    assert(float_equals(output(0, 0), 0.0f));
    assert(float_equals(output(0, 1), 0.0f));
    assert(float_equals(output(0, 2), 0.0f));
    assert(float_equals(output(0, 3), 1.5f));
    assert(float_equals(output(0, 4), 3.0f));

    std::cout << "✓ ReLU 2D ok" << std::endl;
}

void test_relu_4d() {
    std::cout << "\n=== Test: ReLU 4D ===" << std::endl;
    Tensor input(1, 1, 2, 2);
    input(0, 0, 0, 0) = -1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 1, 0) = -3.0f;
    input(0, 0, 1, 1) = 4.0f;

    Tensor output(1, 1, 2, 2);
    ReLULayer layer;
    layer.forward(input, output);

    assert(float_equals(output(0, 0, 0, 0), 0.0f));
    assert(float_equals(output(0, 0, 0, 1), 2.0f));
    assert(float_equals(output(0, 0, 1, 0), 0.0f));
    assert(float_equals(output(0, 0, 1, 1), 4.0f));

    std::cout << "✓ ReLU 4D ok" << std::endl;
}

// =========================
// Sigmoid Tests
// =========================

void test_sigmoid_2d() {
    std::cout << "\n=== Test: Sigmoid 2D ===" << std::endl;
    Tensor input(1, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;

    Tensor output(1, 3);
    SigmoidLayer layer;
    layer.forward(input, output);

    const float e1 = 1.0f / (1.0f + std::exp(1.0f));
    const float e0 = 0.5f;
    const float e2 = 1.0f / (1.0f + std::exp(-1.0f));

    assert(float_equals(output(0, 0), e1));
    assert(float_equals(output(0, 1), e0));
    assert(float_equals(output(0, 2), e2));

    std::cout << "✓ Sigmoid 2D ok" << std::endl;
}

void test_sigmoid_4d() {
    std::cout << "\n=== Test: Sigmoid 4D ===" << std::endl;
    Tensor input(1, 1, 2, 2);
    input(0, 0, 0, 0) = -2.0f;
    input(0, 0, 0, 1) = 0.0f;
    input(0, 0, 1, 0) = 2.0f;
    input(0, 0, 1, 1) = 4.0f;

    Tensor output(1, 1, 2, 2);
    SigmoidLayer layer;
    layer.forward(input, output);

    auto sigmoid = [](float x) {
        return 1.0f / (1.0f + std::exp(-x));
    };

    assert(float_equals(output(0, 0, 0, 0), sigmoid(-2.0f)));
    assert(float_equals(output(0, 0, 0, 1), sigmoid(0.0f)));
    assert(float_equals(output(0, 0, 1, 0), sigmoid(2.0f)));
    assert(float_equals(output(0, 0, 1, 1), sigmoid(4.0f)));

    std::cout << "✓ Sigmoid 4D ok" << std::endl;
}

// =========================
// Tanh Tests
// =========================

void test_tanh_2d() {
    std::cout << "\n=== Test: Tanh 2D ===" << std::endl;
    Tensor input(1, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;

    Tensor output(1, 3);
    TanhLayer layer;
    layer.forward(input, output);

    assert(float_equals(output(0, 0), std::tanh(-1.0f)));
    assert(float_equals(output(0, 1), std::tanh(0.0f)));
    assert(float_equals(output(0, 2), std::tanh(1.0f)));

    std::cout << "✓ Tanh 2D ok" << std::endl;
}

void test_tanh_4d() {
    std::cout << "\n=== Test: Tanh 4D ===" << std::endl;
    Tensor input(1, 1, 2, 2);
    input(0, 0, 0, 0) = -2.0f;
    input(0, 0, 0, 1) = 0.0f;
    input(0, 0, 1, 0) = 2.0f;
    input(0, 0, 1, 1) = 4.0f;

    Tensor output(1, 1, 2, 2);
    TanhLayer layer;
    layer.forward(input, output);

    assert(float_equals(output(0, 0, 0, 0), std::tanh(-2.0f)));
    assert(float_equals(output(0, 0, 0, 1), std::tanh(0.0f)));
    assert(float_equals(output(0, 0, 1, 0), std::tanh(2.0f)));
    assert(float_equals(output(0, 0, 1, 1), std::tanh(4.0f)));

    std::cout << "✓ Tanh 4D ok" << std::endl;
}

// =========================
// Softmax Tests
// =========================

void test_softmax_2d() {
    std::cout << "\n=== Test: Softmax 2D ===" << std::endl;
    Tensor input(2, 3);

    // Row 0: [1, 2, 3]
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    // Row 1: [0, 0, 0]
    input(1, 0) = 0.0f;
    input(1, 1) = 0.0f;
    input(1, 2) = 0.0f;

    Tensor output(2, 3);
    SoftmaxLayer layer;
    layer.forward(input, output);

    // Expected for row 0
    const float max0 = 3.0f;
    const float e0 = std::exp(1.0f - max0);
    const float e1 = std::exp(2.0f - max0);
    const float e2 = std::exp(3.0f - max0);
    const float sum0 = e0 + e1 + e2;

    assert(float_equals(output(0, 0), e0 / sum0));
    assert(float_equals(output(0, 1), e1 / sum0));
    assert(float_equals(output(0, 2), e2 / sum0));

    // Row 1: uniform
    assert(float_equals(output(1, 0), 1.0f / 3.0f));
    assert(float_equals(output(1, 1), 1.0f / 3.0f));
    assert(float_equals(output(1, 2), 1.0f / 3.0f));

    // Check sums ~ 1
    const float sumRow0 = output(0, 0) + output(0, 1) + output(0, 2);
    const float sumRow1 = output(1, 0) + output(1, 1) + output(1, 2);
    assert(float_equals(sumRow0, 1.0f, 1e-4f));
    assert(float_equals(sumRow1, 1.0f, 1e-4f));

    std::cout << "✓ Softmax 2D ok" << std::endl;
}

// =========================
// Main
// =========================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   Activation Layer Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_relu_2d();
    test_relu_4d();
    test_sigmoid_2d();
    test_sigmoid_4d();
    test_tanh_2d();
    test_tanh_4d();
    test_softmax_2d();

    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ All tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
