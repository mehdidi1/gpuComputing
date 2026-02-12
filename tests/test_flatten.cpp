#include <cassert>
#include <cmath>
#include <iostream>
#include "../include/tensor.h"
#include "../include/layers/flatten_layer.h"

static bool float_equals(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_flatten_output_shape() {
    std::cout << "\n=== Test: Flatten Output Shape ===" << std::endl;
    FlattenLayer layer;
    std::vector<int> input_shape = {2, 3, 4, 5};
    auto out_shape = layer.get_output_shape(input_shape);

    assert(out_shape.size() == 2);
    assert(out_shape[0] == 2);
    assert(out_shape[1] == 3 * 4 * 5);

    std::cout << "✓ Output shape correct" << std::endl;
}

void test_flatten_forward_values() {
    std::cout << "\n=== Test: Flatten Forward Values ===" << std::endl;
    Tensor input(1, 2, 2, 3); // B=1, C=2, H=2, W=3

    // Fill with deterministic pattern: value = c*100 + h*10 + w
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 3; ++w) {
                input(0, c, h, w) = static_cast<float>(c * 100 + h * 10 + w);
            }
        }
    }

    FlattenLayer layer;
    auto out_shape = layer.get_output_shape(input.get_shape());
    Tensor output(out_shape[0], out_shape[1]);

    layer.forward(input, output);

    // Expected order: c-major then h then w (matches nested loops in forward)
    int idx = 0;
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 3; ++w) {
                float expected = static_cast<float>(c * 100 + h * 10 + w);
                assert(float_equals(output(0, idx), expected));
                ++idx;
            }
        }
    }

    std::cout << "✓ Flatten values correct" << std::endl;
}

void test_flatten_output_shape_mismatch() {
    std::cout << "\n=== Test: Flatten Output Shape Mismatch ===" << std::endl;
    Tensor input(1, 1, 2, 2); // flattened = 4
    Tensor output(1, 3); // wrong features

    FlattenLayer layer;
    bool threw = false;
    try {
        layer.forward(input, output);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
    std::cout << "✓ Throws on output shape mismatch" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      Flatten Layer Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_flatten_output_shape();
    test_flatten_forward_values();
    test_flatten_output_shape_mismatch();

    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ All tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
