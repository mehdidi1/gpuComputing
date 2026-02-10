// ============================================================================
// Main Application
// ============================================================================
// 

#include "model.h"
#include "layers/conv_layer_cpu.h"
#include "layers/conv_layer_gpu.h"
#include "layers/activation_layers.h"
#include "layers/pooling_layers.h"
#include "layers/flatten_layer.h"
#include "layers/fully_connected_layer.h"
#include "tensor.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstring>
#include <vector>

/**
 * Build SimpleCNN model matching PyTorch architecture:
 * Input: [batch, 1, 28, 28]
 * Conv(32, 5x5, padding=2) -> ReLU -> MaxPool(2x2)
 * Conv(64, 5x5, padding=2) -> ReLU -> MaxPool(2x2)
 * Flatten -> [batch, 3136]
 * FC(3136, 128) -> ReLU
 * FC(128, 10)
 * Output: [batch, 10]
 */
void build_simple_cnn(Model& model)
{
    // Conv1: 32 filters, 5x5 kernel, stride=1, padding=2, in_channels=1
    model.add_layer(std::make_unique<ConvolutionLayerCPU>(32, 5, 5, 1, 2, 1));
    model.add_layer(std::make_unique<ReLULayer>());
    model.add_layer(std::make_unique<MaxPoolingLayer>(2, 2));

    // Conv2: 64 filters, 5x5 kernel, stride=1, padding=2, in_channels=32
    model.add_layer(std::make_unique<ConvolutionLayerCPU>(64, 5, 5, 1, 2, 32));
    model.add_layer(std::make_unique<ReLULayer>());
    model.add_layer(std::make_unique<MaxPoolingLayer>(2, 2));

    // Flatten: [batch, 64, 7, 7] -> [batch, 3136]
    model.add_layer(std::make_unique<FlattenLayer>());

    // FC1: 3136 -> 128
    model.add_layer(std::make_unique<FullyConnectedLayer>(3136, 128));
    model.add_layer(std::make_unique<ReLULayer>());

    // FC2: 128 -> 10
    model.add_layer(std::make_unique<FullyConnectedLayer>(128, 10));
}

/**
 * Load test input from binary file
 * Format: float32 array [num_samples, 1, 28, 28]
 */
std::vector<float> load_binary_data(const std::string& filename, int& num_samples)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> data(file_size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), file_size);

    if (!file.good()) {
        throw std::runtime_error("Error reading input file");
    }

    file.close();

    // Assume format: [num_samples, 1, 28, 28]
    int total_elements = data.size();
    int elements_per_sample = 1 * 28 * 28;
    num_samples = total_elements / elements_per_sample;

    if (total_elements % elements_per_sample != 0) {
        throw std::runtime_error("Input size not divisible by sample size");
    }

    std::cout << "  Loaded " << num_samples << " samples" << std::endl;
    return data;
}

/**
 * Save output to binary file
 * Format: float32 array [num_samples, 10]
 */
void save_binary_output(const std::string& filename, const float* data, int size)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file for writing: " + filename);
    }

    file.write(reinterpret_cast<const char*>(data), size * sizeof(float));

    if (!file.good()) {
        throw std::runtime_error("Error writing output file");
    }

    file.close();
    std::cout << "  Saved " << size << " float32 values" << std::endl;
}

int main(int argc, char* argv[])
{
    std::cout << "C++ CNN Model Validation" << std::endl;
    std::cout << "======================" << std::endl << std::endl;

    try {
        // Get file paths from arguments or use defaults
        std::string weights_file = "python/exported_model/model_weights.bin";
        std::string input_file = "python/exported_model/test_data/test_input.bin";
        std::string output_file = "cpp_output.bin";

        if (argc > 1) weights_file = argv[1];
        if (argc > 2) input_file = argv[2];
        if (argc > 3) output_file = argv[3];

        // ====================================================================
        // 1. Build model architecture
        // ====================================================================
        std::cout << "1. Building SimpleCNN model..." << std::endl;
        Model model;
        build_simple_cnn(model);
        std::cout << "   Model built with " << model.num_layers() << " layers" << std::endl;

        // ====================================================================
        // 2. Load weights
        // ====================================================================
        std::cout << "\n2. Loading weights from " << weights_file << std::endl;
        model.load_weights(weights_file);

        // ====================================================================
        // 3. Load input data
        // ====================================================================
        std::cout << "\n3. Loading test input from " << input_file << std::endl;
        int num_samples = 0;
        std::vector<float> input_data = load_binary_data(input_file, num_samples);

        // ====================================================================
        // 4. Run inference on batches
        // ====================================================================
        std::cout << "\n4. Running inference on " << num_samples << " samples..." << std::endl;

        std::vector<float> all_outputs;
        int batch_size = 32; // Match PyTorch export batch size
        int const_elements_per_sample = 1 * 28 * 28;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - batch_start);

            // Create input tensor [batch_size, 1, 28, 28]
            Tensor input(current_batch_size, 1, 28, 28);
            std::memcpy(input.data(),
                        input_data.data() + batch_start * const_elements_per_sample,
                        current_batch_size * const_elements_per_sample * sizeof(float));

            // Run forward pass
            Tensor output = model.forward(input);

            // Copy output to results
            int output_elements = current_batch_size * 10;
            for (int i = 0; i < output_elements; ++i) {
                all_outputs.push_back(output.data()[i]);
            }

            std::cout << "   Batch " << (batch_start / batch_size + 1)
                      << ": " << current_batch_size << " samples" << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "   Total inference time: " << elapsed.count() << " ms" << std::endl;

        // ====================================================================
        // 5. Save output
        // ====================================================================
        std::cout << "\n5. Saving C++ output to " << output_file << std::endl;
        save_binary_output(output_file, all_outputs.data(), all_outputs.size());

        std::cout << "\n======================" << std::endl;
        std::cout << "✓ Validation complete!" << std::endl;
        std::cout << "\nNext step: Run Python validator to compare outputs" << std::endl;
        std::cout << "======================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return 1;
    }
}
