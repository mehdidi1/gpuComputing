#include <cassert>
#include <iostream>
#include <cmath>
#include "../include/tensor.h"
#include "../include/layers/pooling_layers.h"

// ============================================================================
// Helper Functions
// ============================================================================

bool float_equals(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

// ============================================================================
// MaxPooling Tests
// ============================================================================

void test_maxpool_output_shape() {
    std::cout << "\n=== Test: MaxPooling Output Shape ===" << std::endl;
    MaxPoolingLayer layer(2, 2);
    std::vector<int> input_shape = {1, 3, 4, 4};

    auto output_shape = layer.get_output_shape(input_shape);
    assert(output_shape.size() == 4);
    assert(output_shape[0] == 1);
    assert(output_shape[1] == 3);
    assert(output_shape[2] == 2);
    assert(output_shape[3] == 2);

    std::cout << "✓ Output shape correct: [" << output_shape[0] << ", "
              << output_shape[1] << ", " << output_shape[2] << ", "
              << output_shape[3] << "]" << std::endl;
}

void test_maxpool_values() {
    std::cout << "\n=== Test: MaxPooling Values ===" << std::endl;
    Tensor input(1, 1, 4, 4);

    float vals[4][4] = {
        {1.0f, 2.0f, 5.0f, 6.0f},
        {3.0f, 4.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 13.0f, 14.0f},
        {11.0f, 12.0f, 15.0f, 16.0f}
    };

    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            input(0, 0, h, w) = vals[h][w];
        }
    }

    MaxPoolingLayer layer(2, 2);
    auto out_shape = layer.get_output_shape(input.get_shape());
    Tensor output(out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    layer.forward(input, output);

    assert(float_equals(output(0, 0, 0, 0), 4.0f));
    assert(float_equals(output(0, 0, 0, 1), 8.0f));
    assert(float_equals(output(0, 0, 1, 0), 12.0f));
    assert(float_equals(output(0, 0, 1, 1), 16.0f));

    std::cout << "✓ All max pooling values correct" << std::endl;
}

void test_maxpool_stride_1() {
    std::cout << "\n=== Test: MaxPooling with Stride=1 ===" << std::endl;
    Tensor input(1, 1, 3, 3);

    float vals[3][3] = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };

    for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
            input(0, 0, h, w) = vals[h][w];
        }
    }

    MaxPoolingLayer layer(2, 1);
    auto out_shape = layer.get_output_shape(input.get_shape());
    assert(out_shape[2] == 2 && out_shape[3] == 2);

    Tensor output(out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    layer.forward(input, output);

    assert(float_equals(output(0, 0, 0, 0), 5.0f));
    assert(float_equals(output(0, 0, 0, 1), 6.0f));
    assert(float_equals(output(0, 0, 1, 0), 8.0f));
    assert(float_equals(output(0, 0, 1, 1), 9.0f));

    std::cout << "✓ MaxPooling with stride=1 works correctly" << std::endl;
}

// ============================================================================
// Average Pooling Tests
// ============================================================================

void test_avgpool_output_shape() {
    std::cout << "\n=== Test: AvgPooling Output Shape ===" << std::endl;
    AvgPoolingLayer layer(2, 2);
    std::vector<int> input_shape = {2, 3, 4, 4};

    auto output_shape = layer.get_output_shape(input_shape);
    assert(output_shape.size() == 4);
    assert(output_shape[0] == 2);
    assert(output_shape[1] == 3);
    assert(output_shape[2] == 2);
    assert(output_shape[3] == 2);

    std::cout << "✓ AvgPool output shape correct: [" << output_shape[0] << ", "
              << output_shape[1] << ", " << output_shape[2] << ", "
              << output_shape[3] << "]" << std::endl;
}

void test_avgpool_values() {
    std::cout << "\n=== Test: AvgPooling Values ===" << std::endl;
    Tensor input(1, 1, 4, 4);

    float vals[4][4] = {
        {1.0f, 2.0f, 5.0f, 6.0f},
        {3.0f, 4.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 13.0f, 14.0f},
        {11.0f, 12.0f, 15.0f, 16.0f}
    };

    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            input(0, 0, h, w) = vals[h][w];
        }
    }

    AvgPoolingLayer layer(2, 2);
    auto out_shape = layer.get_output_shape(input.get_shape());
    Tensor output(out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    layer.forward(input, output);

    assert(float_equals(output(0, 0, 0, 0), 2.5f));
    assert(float_equals(output(0, 0, 0, 1), 6.5f));
    assert(float_equals(output(0, 0, 1, 0), 10.5f));
    assert(float_equals(output(0, 0, 1, 1), 14.5f));

    std::cout << "✓ All average pooling values correct" << std::endl;
}

void test_avgpool_multiple_channels() {
    std::cout << "\n=== Test: AvgPooling Multiple Channels ===" << std::endl;
    Tensor input(1, 2, 4, 4);

    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            input(0, 0, h, w) = static_cast<float>(h * 4 + w + 1);
            input(0, 1, h, w) = static_cast<float>(h * 4 + w + 100);
        }
    }

    AvgPoolingLayer layer(2, 2);
    auto out_shape = layer.get_output_shape(input.get_shape());
    Tensor output(out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    layer.forward(input, output);

    assert(float_equals(output(0, 0, 0, 0), 3.5f));
    assert(float_equals(output(0, 1, 0, 0), 102.5f));

    std::cout << "✓ AvgPooling with multiple channels works correctly" << std::endl;
}

void test_avgpool_stride_1() {
    std::cout << "\n=== Test: AvgPooling with Stride=1 ===" << std::endl;
    Tensor input(1, 1, 3, 3);

    float vals[3][3] = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };

    for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
            input(0, 0, h, w) = vals[h][w];
        }
    }

    AvgPoolingLayer layer(2, 1);
    auto out_shape = layer.get_output_shape(input.get_shape());
    assert(out_shape[2] == 2 && out_shape[3] == 2);

    Tensor output(out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    layer.forward(input, output);

    assert(float_equals(output(0, 0, 0, 0), 3.0f));
    assert(float_equals(output(0, 0, 0, 1), 4.0f));
    assert(float_equals(output(0, 0, 1, 0), 6.0f));
    assert(float_equals(output(0, 0, 1, 1), 7.0f));

    std::cout << "✓ AvgPooling with stride=1 works correctly" << std::endl;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "     Pooling Layer Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_maxpool_output_shape();
        test_maxpool_values();
        test_maxpool_stride_1();

        test_avgpool_output_shape();
        test_avgpool_values();
        test_avgpool_multiple_channels();
        test_avgpool_stride_1();

        std::cout << "\n========================================" << std::endl;
        std::cout << "✓ All tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
