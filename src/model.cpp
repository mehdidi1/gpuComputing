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
    // File format (suggested):
    // 1. Header:
    //    - Magic number (e.g., "CNN!")
    //    - Version number
    //    - Number of layers
    //
    // 2. For each layer:
    //    - Layer type ID (enum: CONV, BATCHNORM, FC, etc.)
    //    - Layer-specific parameters (kernel_size, num_filters, etc.)
    //    - Weights dimensions
    //    - Weights data (binary float32)
    //    - Bias dimensions
    //    - Bias data (binary float32)
    //
    // 3. Iterate through layers_ and populate their weights

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

    // Read number of layers
    uint32_t num_layers;
    file.read(reinterpret_cast<char *>(&num_layers), sizeof(uint32_t));
    if (!file.good() || num_layers != layers_.size()) {
        throw std::runtime_error(
            "Layer count mismatch: file has " + std::to_string(num_layers) +
            " but model has " + std::to_string(layers_.size()));
    }

    // Load weights for each layer
    for (size_t i = 0; i < layers_.size(); ++i) {
        // Read layer type
        uint32_t layer_type;
        file.read(reinterpret_cast<char *>(&layer_type), sizeof(uint32_t));
        if (!file.good()) {
            throw std::runtime_error("Error reading layer " +
                                     std::to_string(i) + " type");
        }
        // Read weights dimensions (4D: num_filters, in_channels, height, width)
        uint32_t w_shape[4];
        file.read(reinterpret_cast<char *>(w_shape), 4 * sizeof(uint32_t));
        if (!file.good()) {
            throw std::runtime_error("Error reading weights shape for layer " +
                                     std::to_string(i));
        }

        // Calculate total weights size
        uint32_t weights_size =
            w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3];

        // Read weights data
        std::vector<float> weights_data(weights_size);
        file.read(reinterpret_cast<char *>(weights_data.data()),
                  weights_size * sizeof(float));
        if (!file.good()) {
            throw std::runtime_error("Error reading weights data for layer " +
                                     std::to_string(i));
        }

        // Read bias dimensions
        uint32_t b_shape[1];
        file.read(reinterpret_cast<char *>(b_shape), sizeof(uint32_t));
        if (!file.good()) {
            throw std::runtime_error("Error reading bias shape for layer " +
                                     std::to_string(i));
        }

        // Read bias data
        std::vector<float> bias_data(b_shape[0]);
        file.read(reinterpret_cast<char *>(bias_data.data()),
                  b_shape[0] * sizeof(float));
        if (!file.good()) {
            throw std::runtime_error("Error reading bias data for layer " +
                                     std::to_string(i));
        }

        enum LayerType : uint32_t {
            LAYER_CONV = 0,
            LAYER_FC = 1
        };

        if (layer_type == LAYER_CONV) {
            Tensor kernels(w_shape[0], w_shape[1], w_shape[2], w_shape[3],
                           weights_data.data());

            if (auto *conv_cpu = dynamic_cast<ConvolutionLayerCPU *>(
                    layers_[i].get())) {
                conv_cpu->set_weights(kernels);
                conv_cpu->set_bias(bias_data);
            } else if (auto *conv_gpu = dynamic_cast<ConvolutionLayerGPU *>(
                           layers_[i].get())) {
                conv_gpu->set_weights(kernels);
                conv_gpu->set_bias(bias_data);
            } else {
                throw std::runtime_error(
                    "Layer type mismatch at index " + std::to_string(i) +
                    ": expected ConvolutionLayer");
            }
        } else if (layer_type == LAYER_FC) {
            if (w_shape[2] != 1 || w_shape[3] != 1) {
                throw std::runtime_error(
                    "Invalid FC weight shape at layer " + std::to_string(i) +
                    ": expected [in_features, out_features, 1, 1]");
            }

            Tensor weights_fc(w_shape[0], w_shape[1]);
            if (weights_size !=
                static_cast<uint32_t>(w_shape[0] * w_shape[1])) {
                throw std::runtime_error(
                    "FC weight size mismatch at layer " + std::to_string(i));
            }
            std::memcpy(weights_fc.data(), weights_data.data(),
                        weights_size * sizeof(float));

            if (auto *fc_cpu =
                    dynamic_cast<FullyConnectedLayer *>(layers_[i].get())) {
                fc_cpu->set_weights(weights_fc);
                fc_cpu->set_bias(bias_data);
            } else if (auto *fc_gpu = dynamic_cast<FullyConnectedLayerGPU *>(
                           layers_[i].get())) {
                fc_gpu->set_weights(weights_fc);
                fc_gpu->set_bias(bias_data);
            } else {
                throw std::runtime_error(
                    "Layer type mismatch at index " + std::to_string(i) +
                    ": expected FullyConnectedLayer");
            }
        } else {
            throw std::runtime_error(
                "Unsupported layer type ID " + std::to_string(layer_type) +
                " at index " + std::to_string(i));
        }

        std::cout << "Loaded layer " << i << " with " << weights_size
                  << " weights" << std::endl;
    }

    file.close();
    std::cout << "Successfully loaded weights from " << filename << std::endl;
}

void Model::save_weights(const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weights file for writing: " +
                                 filename);
    }

    auto write_or_throw = [&file](const void *data, std::size_t size,
                                  const std::string &what) {
        file.write(reinterpret_cast<const char *>(data), size);
        if (!file.good()) {
            throw std::runtime_error("Error writing " + what);
        }
    };

    // Header
    const char magic[4] = {'C', 'N', 'N', '!'};
    write_or_throw(magic, 4, "magic number");

    const uint32_t version = 1;
    write_or_throw(&version, sizeof(uint32_t), "version");

    const uint32_t num_layers = static_cast<uint32_t>(layers_.size());
    write_or_throw(&num_layers, sizeof(uint32_t), "num layers");

    enum LayerType : uint32_t {
        LAYER_CONV = 0,
        LAYER_FC = 1
    };

    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto *layer = layers_[i].get();
        if (!layer) {
            throw std::runtime_error("Layer " + std::to_string(i) + " is null");
        }

        if (const auto *conv_cpu =
                dynamic_cast<const ConvolutionLayerCPU *>(layer)) {
            const uint32_t layer_type = LAYER_CONV;
            write_or_throw(&layer_type, sizeof(uint32_t),
                           "layer type (conv)");

            const Tensor &kernels = conv_cpu->get_weights();
            const auto &k_shape = kernels.get_shape();
            if (k_shape.size() != 4) {
                throw std::runtime_error(
                    "Conv weights must be 4D at layer " + std::to_string(i));
            }

            uint32_t w_shape[4] = {static_cast<uint32_t>(k_shape[0]),
                                   static_cast<uint32_t>(k_shape[1]),
                                   static_cast<uint32_t>(k_shape[2]),
                                   static_cast<uint32_t>(k_shape[3])};
            write_or_throw(w_shape, 4 * sizeof(uint32_t),
                           "conv weights shape");

            const uint32_t weights_size =
                w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3];
            write_or_throw(kernels.data(),
                           weights_size * sizeof(float),
                           "conv weights data");

            const auto &bias = conv_cpu->get_bias();
            const uint32_t b_shape[1] = {
                static_cast<uint32_t>(bias.size())};
            write_or_throw(b_shape, sizeof(uint32_t), "conv bias shape");
            if (!bias.empty()) {
                write_or_throw(bias.data(), bias.size() * sizeof(float),
                               "conv bias data");
            }
        } else if (const auto *conv_gpu =
                       dynamic_cast<const ConvolutionLayerGPU *>(layer)) {
            const uint32_t layer_type = LAYER_CONV;
            write_or_throw(&layer_type, sizeof(uint32_t),
                           "layer type (conv)");

            const Tensor &kernels = conv_gpu->get_weights();
            const auto &k_shape = kernels.get_shape();
            if (k_shape.size() != 4) {
                throw std::runtime_error(
                    "Conv weights must be 4D at layer " + std::to_string(i));
            }

            uint32_t w_shape[4] = {static_cast<uint32_t>(k_shape[0]),
                                   static_cast<uint32_t>(k_shape[1]),
                                   static_cast<uint32_t>(k_shape[2]),
                                   static_cast<uint32_t>(k_shape[3])};
            write_or_throw(w_shape, 4 * sizeof(uint32_t),
                           "conv weights shape");

            const uint32_t weights_size =
                w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3];
            write_or_throw(kernels.data(),
                           weights_size * sizeof(float),
                           "conv weights data");

            const auto &bias = conv_gpu->get_bias();
            const uint32_t b_shape[1] = {
                static_cast<uint32_t>(bias.size())};
            write_or_throw(b_shape, sizeof(uint32_t), "conv bias shape");
            if (!bias.empty()) {
                write_or_throw(bias.data(), bias.size() * sizeof(float),
                               "conv bias data");
            }
        } else if (const auto *fc_cpu =
                       dynamic_cast<const FullyConnectedLayer *>(layer)) {
            const uint32_t layer_type = LAYER_FC;
            write_or_throw(&layer_type, sizeof(uint32_t), "layer type (fc)");

            const Tensor &weights = fc_cpu->get_weights();
            const auto &w_shape_raw = weights.get_shape();
            if (w_shape_raw.size() != 2) {
                throw std::runtime_error(
                    "FC weights must be 2D at layer " + std::to_string(i));
            }

            uint32_t w_shape[4] = {static_cast<uint32_t>(w_shape_raw[0]),
                                   static_cast<uint32_t>(w_shape_raw[1]),
                                   1u,
                                   1u};
            write_or_throw(w_shape, 4 * sizeof(uint32_t),
                           "fc weights shape");

            const uint32_t weights_size = w_shape[0] * w_shape[1];
            write_or_throw(weights.data(),
                           weights_size * sizeof(float),
                           "fc weights data");

            const auto &bias = fc_cpu->get_bias();
            const uint32_t b_shape[1] = {
                static_cast<uint32_t>(bias.size())};
            write_or_throw(b_shape, sizeof(uint32_t), "fc bias shape");
            if (!bias.empty()) {
                write_or_throw(bias.data(), bias.size() * sizeof(float),
                               "fc bias data");
            }
        } else if (const auto *fc_gpu =
                       dynamic_cast<const FullyConnectedLayerGPU *>(layer)) {
            const uint32_t layer_type = LAYER_FC;
            write_or_throw(&layer_type, sizeof(uint32_t), "layer type (fc)");

            const Tensor &weights = fc_gpu->get_weights();
            const auto &w_shape_raw = weights.get_shape();
            if (w_shape_raw.size() != 2) {
                throw std::runtime_error(
                    "FC weights must be 2D at layer " + std::to_string(i));
            }

            uint32_t w_shape[4] = {static_cast<uint32_t>(w_shape_raw[0]),
                                   static_cast<uint32_t>(w_shape_raw[1]),
                                   1u,
                                   1u};
            write_or_throw(w_shape, 4 * sizeof(uint32_t),
                           "fc weights shape");

            const uint32_t weights_size = w_shape[0] * w_shape[1];
            write_or_throw(weights.data(),
                           weights_size * sizeof(float),
                           "fc weights data");

            const auto &bias = fc_gpu->get_bias();
            const uint32_t b_shape[1] = {
                static_cast<uint32_t>(bias.size())};
            write_or_throw(b_shape, sizeof(uint32_t), "fc bias shape");
            if (!bias.empty()) {
                write_or_throw(bias.data(), bias.size() * sizeof(float),
                               "fc bias data");
            }
        } else {
            throw std::runtime_error(
                "Unsupported layer for serialization at index " +
                std::to_string(i));
        }
    }

    file.close();
    std::cout << "Successfully saved weights to " << filename << std::endl;
}
