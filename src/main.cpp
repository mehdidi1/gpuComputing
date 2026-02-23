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

static std::vector<float> load_binary_data(const std::string &filename, int &num_samples) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }
    
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

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [cpu|gpu] [num_samples]\n";
    std::cout << "  cpu/gpu: Choose CPU or GPU implementation (default: gpu)\n";
    std::cout << "  num_samples: Number of MNIST samples to process (default: 100, max: 10000)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog_name << " gpu 1000    # Process 1000 samples on GPU\n";
    std::cout << "  " << prog_name << " cpu 100     # Process 100 samples on CPU\n";
}

int main(int argc, char *argv[]) {
    try {
        // Parse command line arguments
        bool use_gpu = true;
        int max_samples = 100;
        
        if (argc > 1) {
            std::string mode = argv[1];
            if (mode == "cpu") {
                use_gpu = false;
            } else if (mode == "gpu") {
                use_gpu = true;
            } else if (mode == "-h" || mode == "--help") {
                print_usage(argv[0]);
                return 0;
            } else {
                std::cerr << "Invalid mode: " << mode << " (use 'cpu' or 'gpu')\n";
                print_usage(argv[0]);
                return 1;
            }
        }
        
        if (argc > 2) {
            max_samples = std::stoi(argv[2]);
            if (max_samples <= 0 || max_samples > 10000) {
                std::cerr << "Invalid number of samples (must be 1-10000)\n";
                return 1;
            }
        }

        std::string weights_file = resolve_path("python/exported_model/model_weights.bin");
        std::string input_file = resolve_path("python/exported_model/test_data/test_input.bin");

        std::cout << "\n========================================\n";
        std::cout << "       CNN MNIST Inference\n";
        std::cout << "========================================\n";
        std::cout << "Mode: " << (use_gpu ? "GPU" : "CPU") << "\n";
        std::cout << "Weights: " << weights_file << "\n";
        std::cout << "Input: " << input_file << "\n";

        // Load data
        int num_samples = 0;
        std::vector<float> input_data = load_binary_data(input_file, num_samples);
        
        num_samples = std::min(num_samples, max_samples);
        std::cout << "Processing " << num_samples << " samples...\n";

        // Build model
        Model model;
        if (use_gpu) {
            model.add_layer(std::make_unique<ConvolutionLayerGPU>(32, 5, 5, 1, 2, 1));
            model.add_layer(std::make_unique<ReLULayer>());
            model.add_layer(std::make_unique<MaxPoolingLayer>(2, 2));
            model.add_layer(std::make_unique<ConvolutionLayerGPU>(64, 5, 5, 1, 2, 32));
            model.add_layer(std::make_unique<ReLULayer>());
            model.add_layer(std::make_unique<MaxPoolingLayer>(2, 2));
            model.add_layer(std::make_unique<FlattenLayer>());
            model.add_layer(std::make_unique<FullyConnectedLayerGPU>(3136, 128));
            model.add_layer(std::make_unique<ReLULayer>());
            model.add_layer(std::make_unique<FullyConnectedLayerGPU>(128, 10));
        } else {
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

        model.load_weights(weights_file);
        std::cout << "Model loaded successfully\n\n";

        // Run inference
        const int batch_size = 32;
        const int elements_per_sample = 1 * 28 * 28;
        int total_batches = (num_samples + batch_size - 1) / batch_size;
        
        std::cout << "Running inference...\n";
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
            int batch_start = batch_idx * batch_size;
            int current_batch_size = std::min(batch_size, num_samples - batch_start);

            Tensor input(current_batch_size, 1, 28, 28);
            std::memcpy(input.data(), 
                        input_data.data() + batch_start * elements_per_sample,
                        static_cast<size_t>(current_batch_size) * elements_per_sample * sizeof(float));

            if (use_gpu) {
                input.to_gpu();
            }

            Tensor output = model.forward(input);
            
            // Show progress every 10 batches
            if ((batch_idx + 1) % 10 == 0 || batch_idx == total_batches - 1) {
                std::cout << "  Processed " << std::min((batch_idx + 1) * batch_size, num_samples) 
                          << "/" << num_samples << " samples\r" << std::flush;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::cout << "\n\n========================================\n";
        std::cout << "            Results\n";
        std::cout << "========================================\n";
        std::cout << "Total samples: " << num_samples << "\n";
        std::cout << "Total time: " << total_time_ms << " ms\n";
        std::cout << "Average per sample: " << (total_time_ms / num_samples) << " ms\n";
        std::cout << "Throughput: " << (num_samples / (total_time_ms / 1000.0)) << " samples/sec\n";
        std::cout << "========================================\n\n";

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
