#include <cassert>
#include <cmath>
#include <iostream>
#include "../include/tensor.h"
#include "../include/layers/fully_connected_layer.h"

static bool float_equals(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

void test_fc_output_shape() {
    std::cout << "\n=== Test: FC Output Shape ===" << std::endl;
    FullyConnectedLayer layer(10, 5);
    std::vector<int> input_shape = {3, 10}; // batch=3, in_features=10
    auto out_shape = layer.get_output_shape(input_shape);

    assert(out_shape.size() == 2);
    assert(out_shape[0] == 3);  // batch unchanged
    assert(out_shape[1] == 5);  // out_features

    std::cout << "✓ Output shape correct" << std::endl;
}

void test_fc_simple_forward() {
    std::cout << "\n=== Test: FC Simple Forward ===" << std::endl;
    
    // Simple 2x2 case for easy verification
    FullyConnectedLayer layer(2, 2);
    
    // Set weights: identity-like matrix
    Tensor weights(2, 2);
    weights(0, 0) = 1.0f;
    weights(0, 1) = 0.0f;
    weights(1, 0) = 0.0f;
    weights(1, 1) = 1.0f;
    layer.set_weights(weights);
    
    // Set bias: [0.5, 1.0]
    std::vector<float> bias = {0.5f, 1.0f};
    layer.set_bias(bias);
    
    // Input: [1, 2] (single batch)
    Tensor input(1, 2);
    input(0, 0) = 3.0f;
    input(0, 1) = 4.0f;
    
    Tensor output(1, 2);
    layer.forward(input, output);
    
    // Expected: [3*1 + 4*0 + 0.5, 3*0 + 4*1 + 1.0] = [3.5, 5.0]
    assert(float_equals(output(0, 0), 3.5f));
    assert(float_equals(output(0, 1), 5.0f));
    
    std::cout << "✓ Simple forward correct" << std::endl;
}

void test_fc_matrix_multiply() {
    std::cout << "\n=== Test: FC Matrix Multiplication ===" << std::endl;
    
    // 3 input features, 2 output features
    FullyConnectedLayer layer(3, 2);
    
    // Weights [3, 2]:
    // [[1, 2],
    //  [3, 4],
    //  [5, 6]]
    Tensor weights(3, 2);
    weights(0, 0) = 1.0f;
    weights(0, 1) = 2.0f;
    weights(1, 0) = 3.0f;
    weights(1, 1) = 4.0f;
    weights(2, 0) = 5.0f;
    weights(2, 1) = 6.0f;
    layer.set_weights(weights);
    
    // Zero bias
    std::vector<float> bias = {0.0f, 0.0f};
    layer.set_bias(bias);
    
    // Input [2, 3]: two samples
    // [[1, 2, 3],
    //  [4, 5, 6]]
    Tensor input(2, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    input(1, 0) = 4.0f;
    input(1, 1) = 5.0f;
    input(1, 2) = 6.0f;
    
    Tensor output(2, 2);
    layer.forward(input, output);
    
    // Expected output[0]: [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    // Expected output[1]: [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
    assert(float_equals(output(0, 0), 22.0f));
    assert(float_equals(output(0, 1), 28.0f));
    assert(float_equals(output(1, 0), 49.0f));
    assert(float_equals(output(1, 1), 64.0f));
    
    std::cout << "✓ Matrix multiplication correct" << std::endl;
}

void test_fc_with_bias() {
    std::cout << "\n=== Test: FC With Bias ===" << std::endl;
    
    FullyConnectedLayer layer(2, 3);
    
    // Weights [2, 3]:
    // [[1, 0, 0],
    //  [0, 1, 0]]
    Tensor weights(2, 3);
    weights(0, 0) = 1.0f;
    weights(0, 1) = 0.0f;
    weights(0, 2) = 0.0f;
    weights(1, 0) = 0.0f;
    weights(1, 1) = 1.0f;
    weights(1, 2) = 0.0f;
    layer.set_weights(weights);
    
    // Bias: [10, 20, 30]
    std::vector<float> bias = {10.0f, 20.0f, 30.0f};
    layer.set_bias(bias);
    
    // Input [1, 2]: [5, 7]
    Tensor input(1, 2);
    input(0, 0) = 5.0f;
    input(0, 1) = 7.0f;
    
    Tensor output(1, 3);
    layer.forward(input, output);
    
    // Expected: [5*1+7*0+10, 5*0+7*1+20, 5*0+7*0+30] = [15, 27, 30]
    assert(float_equals(output(0, 0), 15.0f));
    assert(float_equals(output(0, 1), 27.0f));
    assert(float_equals(output(0, 2), 30.0f));
    
    std::cout << "✓ Bias addition correct" << std::endl;
}

void test_fc_batch_processing() {
    std::cout << "\n=== Test: FC Batch Processing ===" << std::endl;
    
    FullyConnectedLayer layer(2, 1);
    
    // Weights [2, 1]: [[2], [3]]
    Tensor weights(2, 1);
    weights(0, 0) = 2.0f;
    weights(1, 0) = 3.0f;
    layer.set_weights(weights);
    
    // Bias: [1]
    std::vector<float> bias = {1.0f};
    layer.set_bias(bias);
    
    // Input [3, 2]: 3 samples
    Tensor input(3, 2);
    input(0, 0) = 1.0f; input(0, 1) = 1.0f;  // [1, 1]
    input(1, 0) = 2.0f; input(1, 1) = 2.0f;  // [2, 2]
    input(2, 0) = 3.0f; input(2, 1) = 3.0f;  // [3, 3]
    
    Tensor output(3, 1);
    layer.forward(input, output);
    
    // Expected: [1*2+1*3+1, 2*2+2*3+1, 3*2+3*3+1] = [6, 11, 16]
    assert(float_equals(output(0, 0), 6.0f));
    assert(float_equals(output(1, 0), 11.0f));
    assert(float_equals(output(2, 0), 16.0f));
    
    std::cout << "✓ Batch processing correct" << std::endl;
}

void test_fc_shape_mismatch() {
    std::cout << "\n=== Test: FC Shape Mismatch ===" << std::endl;
    
    FullyConnectedLayer layer(3, 2);
    
    // Wrong input features
    Tensor input(1, 5); // should be [1, 3]
    Tensor output(1, 2);
    
    Tensor weights(3, 2);
    weights.zeros();
    layer.set_weights(weights);
    
    bool threw = false;
    try {
        layer.forward(input, output);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    
    assert(threw);
    std::cout << "✓ Throws on shape mismatch" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Fully Connected Layer Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_fc_output_shape();
    test_fc_simple_forward();
    test_fc_matrix_multiply();
    test_fc_with_bias();
    test_fc_batch_processing();

    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ All tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
