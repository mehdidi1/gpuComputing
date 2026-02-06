#pragma once

#include "tensor.h"
#include "layers/layer_interface.h"
#include <vector>
#include <memory>
#include <string>

/**
 * ============================================================================
 * Model / Neural Network Container
 * ============================================================================
 * 
 * Holds a sequence of layers and orchestrates the forward pass
 * 
 * Usage example:
 *   Model model;
 *   model.add_layer(std::make_unique<ConvolutionLayerCPU>(...));
 *   model.add_layer(std::make_unique<ReLULayer>());
 *   model.add_layer(std::make_unique<MaxPoolingLayer>(...));
 *   // ... more layers
 *   
 *   Tensor output = model.forward(input);
 * 
 * TODO: Implement layer sequencing and forward pass orchestration
 */
class Model {
public:
    Model();
    ~Model();
    
    /**
     * Add a layer to the model (takes ownership)
     * 
     * TODO: Store the layer in a vector
     * Validate input/output shapes for compatibility
     * 
     * @param layer - Unique pointer to a layer implementation
     */
    void add_layer(std::unique_ptr<ILayer> layer);
    
    /**
     * Forward pass through all layers
     * 
     * TODO:
     * 1. Start with input tensor
     * 2. For each layer in sequence:
     *    - Get output shape from layer
     *    - Allocate output tensor if needed
     *    - Call layer->forward(input, output)
     *    - Output becomes input for next layer
     * 3. Return final output tensor
     * 
     * @param input - Input tensor
     * @return Final output tensor after all layers
     */
    Tensor forward(const Tensor& input);
    
    /**
     * Get number of layers
     */
    int num_layers() const { return layers_.size(); }
    
    /**
     * Get pointer to layer by index (for debugging/inspection)
     * 
     * @param idx - Layer index
     * @return Raw pointer to layer (valid while model exists)
     */
    ILayer* get_layer(int idx) { return layers_[idx].get(); }
    
    /**
     * Load weights from a file
     * 
     * TODO: Implement weight loading from binary or text format
     * Should iterate through all layers and load their weights
     * 
     * File format (suggested):
     * - Header: Magic number, version, num_layers
     * - For each layer:
     *   - Layer type ID
     *   - Weight dimensions
     *   - Weight data (binary float32)
     *   - Bias dimensions
     *   - Bias data (binary float32)
     * 
     * @param filename - Path to weight file
     */
    void load_weights(const std::string& filename);
    
    /**
     * Save weights to a file
     * 
     * TODO: Implement weight saving for all layers
     * Should use same format as load_weights
     * 
     * @param filename - Path to save weights
     */
    void save_weights(const std::string& filename);
    
private:
    std::vector<std::unique_ptr<ILayer>> layers_;
};
