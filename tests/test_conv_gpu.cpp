#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../include/tensor.h"
#include "../include/layers/conv_layer_cpu.h"
#include "../include/layers/conv_layer_gpu.h"

static bool float_equals(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) < eps;
}

static bool has_cuda_device() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return false;
    }
    return count > 0;
}

static void compare_tensors(const Tensor& a, const Tensor& b, float eps = 1e-3f) {
    const auto shape_a = a.get_shape();
    const auto shape_b = b.get_shape();
    assert(shape_a == shape_b);
    
    // Ensure both tensors have CPU data available
    if (a.is_on_gpu()) {
        const_cast<Tensor&>(a).sync_to_cpu();
    }
    if (b.is_on_gpu()) {
        const_cast<Tensor&>(b).sync_to_cpu();
    }
    
    for (int i = 0; i < a.size(); ++i) {
        assert(float_equals(a.data()[i], b.data()[i], eps));
    }
}

void test_conv_cpu_gpu_simple() {
    std::cout << "\n=== Test: Conv CPU vs GPU (1x1) ===" << std::endl;

    const int B = 1, C = 1, H = 4, W = 4;
    ConvolutionLayerCPU cpu(1, 1, 1, 1, 0, C);
    ConvolutionLayerGPU gpu(1, 1, 1, 1, 0, C);

    Tensor input(B, C, H, W);
    for (int i = 0; i < input.size(); ++i) {
        input.data()[i] = static_cast<float>(i + 1);
    }

    Tensor kernels(1, C, 1, 1);
    kernels(0, 0, 0, 0) = 2.0f;
    std::vector<float> bias = {0.25f};

    cpu.set_weights(kernels);
    cpu.set_bias(bias);
    gpu.set_weights(kernels);
    gpu.set_bias(bias);

    Tensor out_cpu(B, 1, H, W);
    Tensor out_gpu(B, 1, H, W);

    cpu.forward(input, out_cpu);
    gpu.forward(input, out_gpu);

    compare_tensors(out_cpu, out_gpu);
    std::cout << "✓ CPU and GPU outputs match for 1x1 conv" << std::endl;
}

void test_conv_cpu_gpu_3x3() {
    std::cout << "\n=== Test: Conv CPU vs GPU (3x3) ===" << std::endl;

    const int B = 2, C = 2, H = 7, W = 7;
    const int K = 3;
    const int KH = 3, KW = 3;
    const int stride = 1, padding = 1;

    ConvolutionLayerCPU cpu(K, KH, KW, stride, padding, C);
    ConvolutionLayerGPU gpu(K, KH, KW, stride, padding, C);

    Tensor input(B, C, H, W);
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    input(b, c, h, w) = static_cast<float>((b + 1) * 0.1f + (c + 1) * 0.2f + h * 0.3f + w * 0.4f);
                }
            }
        }
    }

    Tensor kernels(K, C, KH, KW);
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    kernels(k, c, kh, kw) = static_cast<float>((k + 1) * 0.05f + (c + 1) * 0.01f + kh * 0.02f + kw * 0.03f);
                }
            }
        }
    }

    std::vector<float> bias(K, 0.1f);

    cpu.set_weights(kernels);
    cpu.set_bias(bias);
    gpu.set_weights(kernels);
    gpu.set_bias(bias);

    const int out_h = (H + 2 * padding - KH) / stride + 1;
    const int out_w = (W + 2 * padding - KW) / stride + 1;

    Tensor out_cpu(B, K, out_h, out_w);
    Tensor out_gpu(B, K, out_h, out_w);

    cpu.forward(input, out_cpu);
    gpu.forward(input, out_gpu);

    compare_tensors(out_cpu, out_gpu, 1e-2f);
    std::cout << "✓ CPU and GPU outputs match for 3x3 conv" << std::endl;
}

int main() {
    std::cout << "==============================================================" << std::endl;
    std::cout << "      CNN Convolution Layer (CPU vs GPU) Test Suite" << std::endl;
    std::cout << "==============================================================" << std::endl;

    if (!has_cuda_device()) {
        std::cout << "No CUDA device detected. Skipping GPU comparison tests." << std::endl;
        return 0;
    }

    try {
        test_conv_cpu_gpu_simple();
        test_conv_cpu_gpu_3x3();

        std::cout << "\n==============================================================" << std::endl;
        std::cout << "         ✓ All CPU vs GPU Convolution Tests Passed" << std::endl;
        std::cout << "==============================================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
