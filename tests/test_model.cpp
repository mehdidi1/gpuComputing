#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>

#include "../include/model.h"
#include "../include/tensor.h"
#include "../include/layers/activation_layers.h"
#include "../include/layers/conv_layer_cpu.h"
#include "../include/layers/fully_connected_layer.h"

static bool float_equals(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static void test_add_null_layer() {
    std::cout << "\n=== Test: Add Null Layer ===" << std::endl;
    Model model;
    std::unique_ptr<ILayer> null_layer;

    bool threw = false;
    try {
        model.add_layer(std::move(null_layer));
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
    std::cout << "✓ Throws on null layer" << std::endl;
}

static void test_forward_empty_model() {
    std::cout << "\n=== Test: Forward Empty Model ===" << std::endl;
    Model model;
    Tensor input(1, 2);

    bool threw = false;
    try {
        (void)model.forward(input);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
    std::cout << "✓ Throws on empty model" << std::endl;
}

static void test_forward_fc_only() {
    std::cout << "\n=== Test: Forward FC Only ===" << std::endl;
    Model model;

    auto fc = std::make_unique<FullyConnectedLayer>(2, 2);
    Tensor weights(2, 2);
    weights(0, 0) = 1.0f; weights(0, 1) = 0.0f;
    weights(1, 0) = 0.0f; weights(1, 1) = 1.0f;
    fc->set_weights(weights);
    fc->set_bias({0.5f, -1.0f});

    model.add_layer(std::move(fc));

    Tensor input(1, 2);
    input(0, 0) = 3.0f;
    input(0, 1) = 4.0f;

    Tensor output = model.forward(input);

    assert(output.ndims() == 2);
    assert(output.batch() == 1);
    assert(output.channels() == 2);
    assert(float_equals(output(0, 0), 3.5f));
    assert(float_equals(output(0, 1), 3.0f));

    std::cout << "✓ FC-only forward produces correct output" << std::endl;
}

static void test_forward_conv_relu() {
    std::cout << "\n=== Test: Forward Conv + ReLU ===" << std::endl;
    Model model;

    auto conv = std::make_unique<ConvolutionLayerCPU>(1, 1, 1, 1, 0, 1);
    Tensor kernels(1, 1, 1, 1);
    kernels(0, 0, 0, 0) = 1.0f;
    conv->set_weights(kernels);
    conv->set_bias({0.0f});

    model.add_layer(std::move(conv));
    model.add_layer(std::make_unique<ReLULayer>());

    Tensor input(1, 1, 2, 2);
    input(0, 0, 0, 0) = -1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 1, 0) = -3.0f;
    input(0, 0, 1, 1) = 4.0f;

    Tensor output = model.forward(input);

    assert(output.ndims() == 4);
    assert(output.batch() == 1);
    assert(output.channels() == 1);
    assert(output.height() == 2);
    assert(output.width() == 2);

    assert(float_equals(output(0, 0, 0, 0), 0.0f));
    assert(float_equals(output(0, 0, 0, 1), 2.0f));
    assert(float_equals(output(0, 0, 1, 0), 0.0f));
    assert(float_equals(output(0, 0, 1, 1), 4.0f));

    std::cout << "✓ Conv+ReLU forward produces correct output" << std::endl;
}

static void test_save_load_conv() {
    std::cout << "\n=== Test: Save/Load Conv Weights ===" << std::endl;
    const std::string filename = "model_conv_weights_test.bin";

    Model model;
    auto conv = std::make_unique<ConvolutionLayerCPU>(1, 1, 1, 1, 0, 1);
    Tensor kernels(1, 1, 1, 1);
    kernels(0, 0, 0, 0) = 2.0f;
    conv->set_weights(kernels);
    conv->set_bias({0.5f});
    model.add_layer(std::move(conv));

    model.save_weights(filename);

    Model loaded;
    loaded.add_layer(std::make_unique<ConvolutionLayerCPU>(1, 1, 1, 1, 0, 1));
    loaded.load_weights(filename);

    Tensor input(1, 1, 2, 2);
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 1, 0) = 3.0f;
    input(0, 0, 1, 1) = 4.0f;

    Tensor out_a = model.forward(input);
    Tensor out_b = loaded.forward(input);

    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 2; ++w) {
            assert(float_equals(out_a(0, 0, h, w), out_b(0, 0, h, w)));
        }
    }

    std::remove(filename.c_str());
    std::cout << "✓ Conv weights round-trip correctly" << std::endl;
}

static void test_save_load_fc() {
    std::cout << "\n=== Test: Save/Load FC Weights ===" << std::endl;
    const std::string filename = "model_fc_weights_test.bin";

    Model model;
    auto fc = std::make_unique<FullyConnectedLayer>(2, 3);
    Tensor weights(2, 3);
    weights(0, 0) = 1.0f; weights(0, 1) = 2.0f; weights(0, 2) = 3.0f;
    weights(1, 0) = 4.0f; weights(1, 1) = 5.0f; weights(1, 2) = 6.0f;
    fc->set_weights(weights);
    fc->set_bias({0.1f, 0.2f, 0.3f});
    model.add_layer(std::move(fc));

    model.save_weights(filename);

    Model loaded;
    loaded.add_layer(std::make_unique<FullyConnectedLayer>(2, 3));
    loaded.load_weights(filename);

    Tensor input(1, 2);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;

    Tensor out_a = model.forward(input);
    Tensor out_b = loaded.forward(input);

    assert(float_equals(out_a(0, 0), out_b(0, 0)));
    assert(float_equals(out_a(0, 1), out_b(0, 1)));
    assert(float_equals(out_a(0, 2), out_b(0, 2)));

    std::remove(filename.c_str());
    std::cout << "✓ FC weights round-trip correctly" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "          Model Unit Tests              " << std::endl;
    std::cout << "========================================" << std::endl;

    test_add_null_layer();
    test_forward_empty_model();
    test_forward_fc_only();
    test_forward_conv_relu();
    test_save_load_conv();
    test_save_load_fc();

    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ All tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
