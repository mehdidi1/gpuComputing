#include <cassert>
#include <cmath>
#include <iostream>
#include "../include/tensor.h"
#include "../include/layers/conv_layer_cpu.h"

static bool float_equals(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

// ============================================================================
// Test 1: Conv Output Shape
// ============================================================================
void test_conv_output_shape() {
    std::cout << "\n=== Test: Conv Output Shape ===" << std::endl;
    
    // Create conv layer: 16 filters, 3x3 kernel, stride=1, padding=0, 3 input channels
    ConvolutionLayerCPU layer(16, 3, 3, 1, 0, 3);
    
    // Input shape: [batch=2, channels=3, height=32, width=32]
    std::vector<int> input_shape = {2, 3, 32, 32};
    auto output_shape = layer.get_output_shape(input_shape);
    
    // Expected output: [2, 16, 30, 30]
    // Formula: (32 + 2*0 - 3) / 1 + 1 = 30
    assert(output_shape.size() == 4);
    assert(output_shape[0] == 2);   // batch unchanged
    assert(output_shape[1] == 16);  // num_filters
    assert(output_shape[2] == 30);  // out_h
    assert(output_shape[3] == 30);  // out_w
    
    std::cout << "✓ Output shape [2, 16, 30, 30] is correct" << std::endl;
}

// ============================================================================
// Test 2: Conv Output Shape with Padding
// ============================================================================
void test_conv_output_shape_with_padding() {
    std::cout << "\n=== Test: Conv Output Shape with Padding ===" << std::endl;
    
    // Create conv layer: 32 filters, 3x3 kernel, stride=1, padding=1, 3 input channels
    ConvolutionLayerCPU layer(32, 3, 3, 1, 1, 3);
    
    // Input shape: [batch=4, channels=3, height=224, width=224]
    std::vector<int> input_shape = {4, 3, 224, 224};
    auto output_shape = layer.get_output_shape(input_shape);
    
    // Expected output: [4, 32, 224, 224]
    // Formula: (224 + 2*1 - 3) / 1 + 1 = 224
    assert(output_shape.size() == 4);
    assert(output_shape[0] == 4);   // batch unchanged
    assert(output_shape[1] == 32);  // num_filters
    assert(output_shape[2] == 224); // out_h
    assert(output_shape[3] == 224); // out_w
    
    std::cout << "✓ Output shape [4, 32, 224, 224] with padding is correct" << std::endl;
}

// ============================================================================
// Test 3: Conv Output Shape with Stride
// ============================================================================
void test_conv_output_shape_with_stride() {
    std::cout << "\n=== Test: Conv Output Shape with Stride ===" << std::endl;
    
    // Create conv layer: 16 filters, 2x2 kernel, stride=2, padding=0, 3 input channels
    ConvolutionLayerCPU layer(16, 2, 2, 2, 0, 3);
    
    // Input shape: [batch=1, channels=3, height=28, width=28]
    std::vector<int> input_shape = {1, 3, 28, 28};
    auto output_shape = layer.get_output_shape(input_shape);
    
    // Expected output: [1, 16, 14, 14]
    // Formula: (28 + 2*0 - 2) / 2 + 1 = 14
    assert(output_shape.size() == 4);
    assert(output_shape[0] == 1);   // batch unchanged
    assert(output_shape[1] == 16);  // num_filters
    assert(output_shape[2] == 14);  // out_h
    assert(output_shape[3] == 14);  // out_w
    
    std::cout << "✓ Output shape [1, 16, 14, 14] with stride is correct" << std::endl;
}

// ============================================================================
// Test 4: Simple Conv Forward Pass - Single Filter
// ============================================================================
void test_conv_simple_forward() {
    std::cout << "\n=== Test: Conv Simple Forward (Single Filter) ===" << std::endl;
    
    // Create a simple 1x1 kernel to just scale the input
    // This eliminates spatial effects and makes verification easy
    ConvolutionLayerCPU layer(1, 1, 1, 1, 0, 1);  // 1 input channel
    
    // Input: [batch=1, channels=1, height=3, width=3]
    Tensor input(1, 1, 3, 3);
    for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
            input(0, 0, h, w) = static_cast<float>(h * 3 + w + 1); // [1,2,3; 4,5,6; 7,8,9]
        }
    }
    
    // Set kernel = [[[[2.0]]]] (scale by 2)
    Tensor kernels(1, 1, 1, 1);
    kernels(0, 0, 0, 0) = 2.0f;
    layer.set_weights(kernels);
    
    // Set bias = [0.5]
    std::vector<float> bias = {0.5f};
    layer.set_bias(bias);
    
    // Forward pass
    Tensor output(1, 1, 3, 3);
    layer.forward(input, output);
    
    // Expected output: input * 2 + 0.5 = [2.5, 4.5, 6.5; 8.5, 10.5, 12.5; 14.5, 16.5, 18.5]
    for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
            float expected = input(0, 0, h, w) * 2.0f + 0.5f;
            float actual = output(0, 0, h, w);
            assert(float_equals(actual, expected));
        }
    }
    
    std::cout << "✓ Simple 1x1 convolution output is correct" << std::endl;
}

// ============================================================================
// Test 5: Conv with 3x3 Kernel
// ============================================================================
void test_conv_3x3_kernel() {
    std::cout << "\n=== Test: Conv with 3x3 Kernel ===" << std::endl;
    
    // Create conv layer: 1 filter, 3x3 kernel, stride=1, padding=0, 1 input channel
    ConvolutionLayerCPU layer(1, 3, 3, 1, 0, 1);
    
    // Input: [batch=1, channels=1, height=5, width=5]
    Tensor input(1, 1, 5, 5);
    for (int i = 0; i < 25; ++i) {
        input(0, 0, i / 5, i % 5) = static_cast<float>(i);
    }
    
    // Kernel: All ones (simple sum filter)
    Tensor kernels(1, 1, 3, 3);
    for (int i = 0; i < 9; ++i) {
        kernels(0, 0, i / 3, i % 3) = 1.0f;
    }
    layer.set_weights(kernels);
    
    // No bias
    std::vector<float> bias = {0.0f};
    layer.set_bias(bias);
    
    // Forward pass
    Tensor output(1, 1, 3, 3);
    layer.forward(input, output);
    
    // Top-left corner [0:3, 0:3] = [0,1,2; 5,6,7; 10,11,12], sum=54
    float expected_00 = 0.0f + 1.0f + 2.0f + 5.0f + 6.0f + 7.0f + 10.0f + 11.0f + 12.0f;
    assert(float_equals(output(0, 0, 0, 0), expected_00));
    
    std::cout << "✓ 3x3 kernel convolution output is correct" << std::endl;
}

// ============================================================================
// Test 6: Conv Batch Processing
// ============================================================================
void test_conv_batch_processing() {
    std::cout << "\n=== Test: Conv Batch Processing ===" << std::endl;
    
    // Create conv layer: 2 filters, 1x1 kernel, 2 input channels
    ConvolutionLayerCPU layer(2, 1, 1, 1, 0, 2);
    
    // Input: [batch=3, channels=2, height=2, width=2]
    Tensor input(3, 2, 2, 2);
    for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    input(b, c, h, w) = static_cast<float>(b * 100 + c * 10 + h * 2 + w);
                }
            }
        }
    }
    
    // Kernels: identity-like (filter 0: [1,0], filter 1: [0,1])
    Tensor kernels(2, 2, 1, 1);
    kernels(0, 0, 0, 0) = 1.0f; // filter 0, channel 0
    kernels(0, 1, 0, 0) = 0.0f; // filter 0, channel 1
    kernels(1, 0, 0, 0) = 0.0f; // filter 1, channel 0
    kernels(1, 1, 0, 0) = 1.0f; // filter 1, channel 1
    layer.set_weights(kernels);
    
    // Biases: [10.0, 20.0]
    std::vector<float> bias = {10.0f, 20.0f};
    layer.set_bias(bias);
    
    // Forward pass
    Tensor output(3, 2, 2, 2);
    layer.forward(input, output);
    
    // For batch 0, channel 0: input(0,0,0,0)=0, input(0,1,0,0)=10
    // Filter 0 output = 0*1 + 10*0 + 10 = 10
    // Filter 1 output = 0*0 + 10*1 + 20 = 30
    assert(float_equals(output(0, 0, 0, 0), 10.0f)); // filter 0
    assert(float_equals(output(0, 1, 0, 0), 30.0f)); // filter 1
    
    std::cout << "✓ Batch processing produces correct results for all samples" << std::endl;
}

// ============================================================================
// Test 7: Conv with Multi-Channel Input
// ============================================================================
void test_conv_multichannel() {
    std::cout << "\n=== Test: Conv with Multi-Channel Input ===" << std::endl;
    
    // Create conv layer: 1 filter, 1x1 kernel, 3 input channels
    ConvolutionLayerCPU layer(1, 1, 1, 1, 0, 3);
    
    // Input: [batch=1, channels=3, height=2, width=2]
    Tensor input(1, 3, 2, 2);
    // Channel 0: [[1, 2], [3, 4]]
    // Channel 1: [[5, 6], [7, 8]]
    // Channel 2: [[9, 10], [11, 12]]
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                input(0, c, h, w) = static_cast<float>(c * 8 + h * 2 + w + 1);
            }
        }
    }
    
    // Kernel: [1.0, 2.0, 3.0] for each channel
    Tensor kernels(1, 3, 1, 1);
    kernels(0, 0, 0, 0) = 1.0f;
    kernels(0, 1, 0, 0) = 2.0f;
    kernels(0, 2, 0, 0) = 3.0f;
    layer.set_weights(kernels);
    
    // No bias
    std::vector<float> bias = {0.0f};
    layer.set_bias(bias);
    
    // Forward pass
    Tensor output(1, 1, 2, 2);
    layer.forward(input, output);
    
    // At position (0,0): 1*1 + 5*2 + 9*3 = 1 + 10 + 27 = 38
    float expected_00 = input(0, 0, 0, 0) * 1.0f + input(0, 1, 0, 0) * 2.0f + input(0, 2, 0, 0) * 3.0f;
    assert(float_equals(output(0, 0, 0, 0), expected_00));
    
    std::cout << "✓ Multi-channel convolution output is correct" << std::endl;
}

// ============================================================================
// Test 8: Conv Invalid Channel Mismatch
// ============================================================================
void test_conv_invalid_input_shape() {
    std::cout << "\n=== Test: Conv Invalid Channel Mismatch ===" << std::endl;
    
    // Layer expects 3 input channels
    ConvolutionLayerCPU layer(16, 3, 3, 1, 0, 3);
    
    // Setup with wrong number of channels
    Tensor input(1, 5, 32, 32);  // 5 channels instead of 3
    input.zeros();
    
    Tensor kernels(16, 3, 3, 3);
    kernels.zeros();
    layer.set_weights(kernels);
    
    std::vector<float> bias(16, 0.0f);
    layer.set_bias(bias);
    
    Tensor output(1, 16, 30, 30);
    
    bool caught_exception = false;
    try {
        layer.forward(input, output);
    } catch (const std::runtime_error& e) {
        caught_exception = true;
    }
    
    assert(caught_exception);
    std::cout << "✓ Channel mismatch correctly detected and rejected" << std::endl;
}

// ============================================================================
// Test 9: Conv Invalid Output Shape
// ============================================================================
void test_conv_invalid_output_shape() {
    std::cout << "\n=== Test: Conv Invalid Output Shape ===" << std::endl;
    
    ConvolutionLayerCPU layer(16, 3, 3, 1, 0, 3);
    
    // Setup weights and bias
    Tensor input(1, 3, 32, 32);
    input.zeros();
    
    Tensor kernels(16, 3, 3, 3);
    kernels.zeros();
    layer.set_weights(kernels);
    
    std::vector<float> bias(16, 0.0f);
    layer.set_bias(bias);
    
    // Try wrong output shape
    Tensor output_wrong(1, 32, 30, 30);  // wrong dimensions
    
    bool caught_exception = false;
    try {
        layer.forward(input, output_wrong);
    } catch (const std::runtime_error& e) {
        caught_exception = true;
    }
    
    assert(caught_exception);
    std::cout << "✓ Invalid output shape correctly rejected" << std::endl;
}

// ============================================================================
// Test 10: Conv with Padding
// ============================================================================
void test_conv_with_padding() {
    std::cout << "\n=== Test: Conv with Padding ===" << std::endl;
    
    // Create conv layer: 1 filter, 3x3 kernel, padding=1, 1 input channel
    ConvolutionLayerCPU layer(1, 3, 3, 1, 1, 1);
    
    // Input: [batch=1, channels=1, height=3, width=3]
    Tensor input(1, 1, 3, 3);
    for (int i = 0; i < 9; ++i) {
        input(0, 0, i / 3, i % 3) = static_cast<float>(i + 1);
    }
    
    // Kernel: All ones
    Tensor kernels(1, 1, 3, 3);
    kernels.zeros();
    for (int i = 0; i < 9; ++i) {
        kernels(0, 0, i / 3, i % 3) = 1.0f;
    }
    layer.set_weights(kernels);
    
    // No bias
    std::vector<float> bias = {0.0f};
    layer.set_bias(bias);
    
    // Forward pass
    Tensor output(1, 1, 3, 3);
    layer.forward(input, output);
    
    // With padding=1, spatial dimensions preserved
    // Top-left corner includes zero-padded region
    // Valid input region [0:3, 0:3] = 1+2+4+5 = 12
    float expected_00 = 1.0f + 2.0f + 4.0f + 5.0f;  // corner pixels only
    assert(float_equals(output(0, 0, 0, 0), expected_00));
    
    std::cout << "✓ Padding correctly applied at boundaries" << std::endl;
}

// ============================================================================
// Main Test Runner
// ============================================================================
int main() {
    std::cout << "==============================================================" << std::endl;
    std::cout << "         CNN Convolution Layer (CPU) Test Suite" << std::endl;
    std::cout << "==============================================================" << std::endl;
    
    try {
        test_conv_output_shape();
        test_conv_output_shape_with_padding();
        test_conv_output_shape_with_stride();
        test_conv_simple_forward();
        test_conv_3x3_kernel();
        test_conv_batch_processing();
        test_conv_multichannel();
        test_conv_with_padding();
        
        std::cout << "\n==============================================================" << std::endl;
        std::cout << "              ✓ All Convolution Tests Passed" << std::endl;
        std::cout << "==============================================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
