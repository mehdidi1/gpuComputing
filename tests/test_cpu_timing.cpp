#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
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
    if (f.good()) return path;
    return "../" + path;
}

static std::vector<float> load_binary_data(const std::string &filename,
                                           int &num_samples) {
    std::ifstream file(filename, std::ios::binary);
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> data(file_size / sizeof(float));
    file.read(reinterpret_cast<char *>(data.data()), file_size);
    file.close();

    int total_elements = static_cast<int>(data.size());
    int elements_per_sample = 1 * 28 * 28;
    num_samples = total_elements / elements_per_sample;
    return data;
}

int main(int argc, char *argv[]) {
    try {
        std::string weights_file = resolve_path("python/exported_model/model_weights.bin");
        std::string input_file = resolve_path("python/exported_model/test_data/test_input.bin");

        int num_samples = 0;
        std::vector<float> input_data = load_binary_data(input_file, num_samples);

        // Build model with CPU layers (same architecture as GPU model)
        std::vector<std::unique_ptr<ILayer>> layers;
        layers.push_back(std::make_unique<ConvolutionLayerCPU>(32, 5, 5, 1, 2, 1));
        layers.push_back(std::make_unique<ReLULayer>());
        layers.push_back(std::make_unique<MaxPoolingLayer>(2, 2));
        layers.push_back(std::make_unique<ConvolutionLayerCPU>(64, 5, 5, 1, 2, 32));
        layers.push_back(std::make_unique<ReLULayer>());
        layers.push_back(std::make_unique<MaxPoolingLayer>(2, 2));
        layers.push_back(std::make_unique<FlattenLayer>());
        layers.push_back(std::make_unique<FullyConnectedLayer>(3136, 128));
        layers.push_back(std::make_unique<ReLULayer>());
        layers.push_back(std::make_unique<FullyConnectedLayer>(128, 10));

        // Load weights
        Model model;
        for (auto &layer : layers) {
            model.add_layer(std::move(layer));
        }
        model.load_weights(weights_file);

        std::cout << "\n=== Layer-wise Timing (CPU) ===" << std::endl;

        const int batch_size = 32;
        const int elements_per_sample = 1 * 28 * 28;

        double layer_times[10] = {0};
        const char *layer_names[] = {
            "Conv2d(1->32)",    // 0
            "ReLU",             // 1
            "MaxPool",          // 2
            "Conv2d(32->64)",   // 3
            "ReLU",             // 4
            "MaxPool",          // 5
            "Flatten",          // 6
            "FC(3136->128)",    // 7
            "ReLU",             // 8
            "FC(128->10)"       // 9
        };

        // Run one batch through all layers, timing each
        Tensor input(batch_size, 1, 28, 28);
        std::memcpy(input.data(), input_data.data(),
                    static_cast<size_t>(batch_size) * elements_per_sample * sizeof(float));

        Tensor current = input.clone();

        for (size_t l = 0; l < model.num_layers(); ++l) {
            std::vector<int> out_shape = model.get_layer(l)->get_output_shape(current.get_shape());
            Tensor output = (out_shape.size() == 4)
                                ? Tensor(out_shape[0], out_shape[1], out_shape[2], out_shape[3])
                                : Tensor(out_shape[0], out_shape[1]);

            auto start = std::chrono::high_resolution_clock::now();
            model.get_layer(l)->forward(current, output);
            auto end = std::chrono::high_resolution_clock::now();

            layer_times[l] = std::chrono::duration<double, std::milli>(end - start).count();
            current = std::move(output);

            std::cout << "Layer " << l << " (" << layer_names[l] << "): " << layer_times[l]
                      << " ms" << std::endl;
        }

        double total = 0;
        for (int i = 0; i < 10; ++i) total += layer_times[i];
        std::cout << "\nTotal: " << total << " ms\n" << std::endl;

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
