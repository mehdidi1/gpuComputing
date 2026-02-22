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
 */
class Model {
public:
    Model();
    ~Model();
    
    /**
     * Add a layer to the model (takes ownership)
     * 
     * @param layer - Unique pointer to a layer implementation
     */
    void add_layer(std::unique_ptr<ILayer> layer);
    
    /**
     * Forward pass through all layers
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

     * 
     * File format:
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
    
private:
    std::vector<std::unique_ptr<ILayer>> layers_;
};
