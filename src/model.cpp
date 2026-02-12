// ============================================================================
// Model Implementation
// ============================================================================

#include "model.h"
#include "layers/conv_layer_cpu.h"
#include "layers/conv_layer_gpu.h"
#include "layers/fully_connected_layer.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

Model::Model() {}

Model::~Model() {}

void Model::add_layer(std::unique_ptr<ILayer> layer) {
    if (!layer) {
        throw std::runtime_error("Cannot add null layer");
    }

    // Add layer to the vector
    layers_.push_back(std::move(layer));
}

Tensor Model::forward(const Tensor &input) {
    if (layers_.empty()) {
        throw std::runtime_error("Model has no layers");
    }

    Tensor current_tensor = input.clone();

    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto &layer = layers_[i];

        if (!layer) {
            throw std::runtime_error("Layer " + std::to_string(i) + " is null");
        }

        // Get output shape for this layer
        std::vector<int> output_shape =
            layer->get_output_shape(current_tensor.get_shape());

        if (output_shape.size() != 4 && output_shape.size() != 2) {
            throw std::runtime_error("Invalid output shape from layer " +
                                     std::to_string(i));
        }

        // Create output tensor with correct shape
        Tensor output_tensor = (output_shape.size() == 4)
                                   ? Tensor(output_shape[0], output_shape[1],
                                            output_shape[2], output_shape[3])
                                   : Tensor(output_shape[0], output_shape[1]);

        // Execute forward pass through this layer
        layer->forward(current_tensor, output_tensor);

        // Update current tensor for next layer
        current_tensor = output_tensor.clone();
    }

    return current_tensor;
}

void Model::load_weights(const std::string &filename) {
    // File format:
    // 1. Header:
    //    - Magic number (e.g., "CNN!")
    //    - Version number
    //    - Number of weight-bearing layers (Conv, FC only)
    //
    // 2. For each weight layer:
    //    - Layer type ID (enum: CONV=0, FC=1)
    //    - Weights dimensions
    //    - Weights data (binary float32)
    //    - Bias dimensions
    //    - Bias data (binary float32)
    //
    // 3. Load into model layers that have weights, in order

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weights file: " + filename);
    }

    // Read magic number
    char magic[4];
    file.read(magic, 4);
    if (!file.good() || std::string(magic, 4) != "CNN!") {
        throw std::runtime_error(
            "Invalid weights file format (magic number mismatch)");
    }

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
    if (!file.good() || version != 1) {
        throw std::runtime_error("Unsupported weights file version: " +
                                 std::to_string(version));
    }

    // Read number of weight layers
    uint32_t num_weight_layers;
    file.read(reinterpret_cast<char *>(&num_weight_layers), sizeof(uint32_t));
    if (!file.good()) {
        throw std::runtime_error("Error reading number of weight layers");
    }

    enum LayerType : uint32_t {
        LAYER_CONV = 0,
        LAYER_FC = 1
    };

    // Find all weight-bearing layers in the model
    std::vector<size_t> weight_layer_indices;
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (dynamic_cast<ConvolutionLayerCPU *>(layers_[i].get()) ||
            dynamic_cast<ConvolutionLayerGPU *>(layers_[i].get()) ||
            dynamic_cast<FullyConnectedLayer *>(layers_[i].get()) ||
            dynamic_cast<FullyConnectedLayerGPU *>(layers_[i].get())) {
            weight_layer_indices.push_back(i);
        }
    }

    if (weight_layer_indices.size() !=
        static_cast<size_t>(num_weight_layers)) {
        throw std::runtime_error(
            "Weight layer count mismatch: file has " +
            std::to_string(num_weight_layers) + " but model has " +
            std::to_string(weight_layer_indices.size()));
    }

    // Load weights for each weight layer
    for (size_t w_idx = 0; w_idx < num_weight_layers; ++w_idx) {
        size_t layer_idx = weight_layer_indices[w_idx];

        // Read layer type
        uint32_t layer_type;
        file.read(reinterpret_cast<char *>(&layer_type), sizeof(uint32_t));
        if (!file.good()) {
            throw std::runtime_error("Error reading weight layer " +
                                     std::to_string(w_idx) + " type");
        }

        // Read weights dimensions
        uint32_t w_shape[4];
        file.read(reinterpret_cast<char *>(w_shape), 4 * sizeof(uint32_t));
        if (!file.good()) {
            throw std::runtime_error(
                "Error reading weights shape for weight layer " +
                std::to_string(w_idx));
        }

        // Calculate total weights size
        uint32_t weights_size =
            w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3];

        // Read weights data
        std::vector<float> weights_data(weights_size);
        file.read(reinterpret_cast<char *>(weights_data.data()),
                  weights_size * sizeof(float));
        if (!file.good()) {
            throw std::runtime_error(
                "Error reading weights data for weight layer " +
                std::to_string(w_idx));
        }

        // Read bias dimensions
        uint32_t b_shape[1];
        file.read(reinterpret_cast<char *>(b_shape), sizeof(uint32_t));
        if (!file.good()) {
            throw std::runtime_error(
                "Error reading bias shape for weight layer " +
                std::to_string(w_idx));
        }

        // Read bias data
        std::vector<float> bias_data(b_shape[0]);
        file.read(reinterpret_cast<char *>(bias_data.data()),
                  b_shape[0] * sizeof(float));
        if (!file.good()) {
            throw std::runtime_error(
                "Error reading bias data for weight layer " +
                std::to_string(w_idx));
        }

        if (layer_type == LAYER_CONV) {
            Tensor kernels(w_shape[0], w_shape[1], w_shape[2], w_shape[3],
                           weights_data.data());

            if (auto *conv_cpu = dynamic_cast<ConvolutionLayerCPU *>(
                    layers_[layer_idx].get())) {
                conv_cpu->set_weights(kernels);
                conv_cpu->set_bias(bias_data);
            } else if (auto *conv_gpu = dynamic_cast<ConvolutionLayerGPU *>(
                           layers_[layer_idx].get())) {
                conv_gpu->set_weights(kernels);
                conv_gpu->set_bias(bias_data);
            } else {
                throw std::runtime_error(
                    "Layer type mismatch at weight layer " +
                    std::to_string(w_idx) + " (model layer " +
                    std::to_string(layer_idx) +
                    "): expected ConvolutionLayer");
            }
        } else if (layer_type == LAYER_FC) {
            if (w_shape[2] != 1 || w_shape[3] != 1) {
                throw std::runtime_error(
                    "Invalid FC weight shape at weight layer " +
                    std::to_string(w_idx) + ": expected [in_features, out_features, 1, 1]");
            }

            Tensor weights_fc(w_shape[0], w_shape[1]);
            if (weights_size !=
                static_cast<uint32_t>(w_shape[0] * w_shape[1])) {
                throw std::runtime_error(
                    "FC weight size mismatch at weight layer " +
                    std::to_string(w_idx));
            }
            std::memcpy(weights_fc.data(), weights_data.data(),
                        weights_size * sizeof(float));

            if (auto *fc_cpu =
                    dynamic_cast<FullyConnectedLayer *>(
                        layers_[layer_idx].get())) {
                fc_cpu->set_weights(weights_fc);
                fc_cpu->set_bias(bias_data);
            } else if (auto *fc_gpu = dynamic_cast<FullyConnectedLayerGPU *>(
                           layers_[layer_idx].get())) {
                fc_gpu->set_weights(weights_fc);
                fc_gpu->set_bias(bias_data);
            } else {
                throw std::runtime_error(
                    "Layer type mismatch at weight layer " +
                    std::to_string(w_idx) + " (model layer " +
                    std::to_string(layer_idx) +
                    "): expected FullyConnectedLayer");
            }
        } else {
            throw std::runtime_error(
                "Unsupported layer type ID " + std::to_string(layer_type) +
                " at weight layer " + std::to_string(w_idx));
        }

        std::cout << "Loaded weight layer " << w_idx << " into model layer "
                  << layer_idx << std::endl;
    }

    file.close();
    std::cout << "Successfully loaded weights from " << filename << std::endl;
}

