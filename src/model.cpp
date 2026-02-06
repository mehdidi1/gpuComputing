// ============================================================================
// Model Implementation
// ============================================================================
// 
// TODO: Implement the layer sequencing and orchestration
// 

#include "model.h"
#include <algorithm>

Model::Model() {}

Model::~Model() {}

void Model::add_layer(std::unique_ptr<ILayer> layer)
{
    // TODO: Add layer to the vector
    // Can optionally validate shape compatibility:
    // - Get output shape of previous layer (if exists)
    // - Get output shape of this layer with same input
    // - Check if they're compatible
}

Tensor Model::forward(const Tensor& input)
{
    // TODO: Execute forward pass through all layers
    // 
    // Algorithm:
    // 1. Start with current_tensor = input
    // 2. For each layer:
    //    a. Get output shape using layer->get_output_shape(...)
    //    b. Create output tensor with that shape
    //    c. Call layer->forward(current_tensor, output_tensor)
    //    d. current_tensor = output_tensor
    // 3. Return final output_tensor
    throw std::runtime_error("Not implemented");
}

void Model::load_weights(const std::string& filename)
{
    // TODO: Load weights from binary file
    // 
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
    // 
    // Error handling:
    // - Check file exists
    // - Validate magic number
    // - Validate version compatibility
    // - Check dimensions match
    throw std::runtime_error("Not implemented");
}

void Model::save_weights(const std::string& filename)
{
    // TODO: Save weights to binary file (inverse of load_weights)
    // Write same format as load_weights expects
    throw std::runtime_error("Not implemented");
}
