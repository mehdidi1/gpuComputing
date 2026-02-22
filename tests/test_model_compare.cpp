#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/model.h"
#include "../include/tensor.h"
#include "../include/layers/activation_layers.h"
#include "../include/layers/conv_layer_cpu.h"
#include "../include/layers/conv_layer_gpu.h"
#include "../include/layers/flatten_layer.h"
#include "../include/layers/fully_connected_layer.h"
#include "../include/layers/pooling_layers.h"

static std::string resolve_path(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (f.good()) {
        return path;
    }
    std::string up_one = "../" + path;
    std::ifstream f2(up_one, std::ios::binary);
    if (f2.good()) {
        return up_one;
    }
    return path; // will fail later with clear error
}

static std::vector<float> load_binary_data(const std::string &filename,
                                           int &num_samples) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> data(file_size / sizeof(float));
    file.read(reinterpret_cast<char *>(data.data()), file_size);

    if (!file.good()) {
        throw std::runtime_error("Error reading input file");
    }

    file.close();

    int total_elements = static_cast<int>(data.size());
    int elements_per_sample = 1 * 28 * 28;
    num_samples = total_elements / elements_per_sample;

    if (total_elements % elements_per_sample != 0) {
        throw std::runtime_error("Input size not divisible by sample size");
    }

    return data;
}

static void build_simple_cnn_cpu(Model &model) {
    model.add_layer(std::make_unique<ConvolutionLayerCPU>(32, 5, 5, 1, 2, 1));
    model.add_layer(std::make_unique<ReLULayer>());
    model.add_layer(std::make_unique<MaxPoolingLayer>(2, 2));

    model.add_layer(std::make_unique<ConvolutionLayerCPU>(64, 5, 5, 1, 2, 32));
    model.add_layer(std::make_unique<ReLULayer>());
    model.add_layer(std::make_unique<MaxPoolingLayer>(2, 2));

    model.add_layer(std::make_unique<FlattenLayer>());

    model.add_layer(std::make_unique<FullyConnectedLayer>(3136, 128));
    model.add_layer(std::make_unique<ReLULayer>());

    model.add_layer(std::make_unique<FullyConnectedLayer>(128, 10));
}

static void build_simple_cnn_gpu(Model &model) {
    model.add_layer(std::make_unique<ConvolutionLayerGPU>(32, 5, 5, 1, 2, 1));
    model.add_layer(std::make_unique<ReLULayer>());
    model.add_layer(std::make_unique<MaxPoolingLayer>(2, 2));

    model.add_layer(std::make_unique<ConvolutionLayerGPU>(64, 5, 5, 1, 2, 32));
    model.add_layer(std::make_unique<ReLULayer>());
    model.add_layer(std::make_unique<MaxPoolingLayer>(2, 2));

    model.add_layer(std::make_unique<FlattenLayer>());

    // Use GPU FC layers
    model.add_layer(std::make_unique<FullyConnectedLayerGPU>(3136, 128));
    model.add_layer(std::make_unique<ReLULayer>());

    model.add_layer(std::make_unique<FullyConnectedLayerGPU>(128, 10));
}

static double time_forward_ms(Model &model, const Tensor &input, Tensor &output) {
    auto start = std::chrono::high_resolution_clock::now();
    output = model.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char *argv[]) {
    try {
        std::string weights_file = "python/exported_model/model_weights.bin";
        std::string input_file = "python/exported_model/test_data/test_input.bin";

        if (argc > 1) weights_file = argv[1];
        if (argc > 2) input_file = argv[2];

        weights_file = resolve_path(weights_file);
        input_file = resolve_path(input_file);

        std::cout << "CPU vs GPU Model Comparison" << std::endl;
        std::cout << "Weights: " << weights_file << std::endl;
        std::cout << "Input:   " << input_file << std::endl;

        int num_samples = 0;
        std::vector<float> input_data = load_binary_data(input_file, num_samples);

        Model gpu_model;

        build_simple_cnn_gpu(gpu_model);
        gpu_model.load_weights(weights_file);

        const int batch_size = 32;
        const int elements_per_sample = 1 * 28 * 28;

        double gpu_total_ms = 0.0;

        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            const int current_batch = std::min(batch_size, num_samples - batch_start);

            Tensor input(current_batch, 1, 28, 28);
            std::memcpy(input.data(),
                        input_data.data() + batch_start * elements_per_sample,
                        static_cast<size_t>(current_batch) * elements_per_sample * sizeof(float));

            Tensor out_cpu(1, 1);
            Tensor out_gpu(1, 1);
            gpu_total_ms += time_forward_ms(gpu_model, input, out_gpu);
        }
        std::cout << "GPU time (ms): " << gpu_total_ms << std::endl;

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
