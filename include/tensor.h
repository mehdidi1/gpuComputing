#pragma once

#include <vector>
#include <memory>
#include <cstring>
#include <stdexcept>

/**
 * ============================================================================
 * Tensor Class - Multi-dimensional array container
 * ============================================================================
 * 
 * Represents n-dimensional arrays (4D for images: batch, channels, height, width)
 * Manages memory allocation, deallocation, and element access.
 * 
 */
class Tensor {
public:
    // ========================================================================
    // CONSTRUCTORS & DESTRUCTORS
    // ========================================================================
    
    /**
     * Constructor for 4D tensors (batch, channels, height, width)
     * Allocates memory for the entire tensor, initialized to zeros
     * 
     * @param batch   - Number of samples in batch
     * @param channels - Number of channels/filters
     * @param height  - Height of each sample
     * @param width   - Width of each sample
     */
    Tensor(int batch, int channels, int height, int width);
    
    /**
     * Constructor for 2D tensors (batch, features)
     * Used for flattened layers and fully-connected inputs
     * 
     * @param batch    - Number of samples
     * @param features - Number of features per sample
     */
    Tensor(int batch, int features);
    
    /**
     * Constructor that initializes from raw float pointer
     * Copies data from external source into internal storage
     * 
     * @param batch    - Batch size
     * @param channels - Channels
     * @param height   - Height
     * @param width    - Width
     * @param data     - Pointer to float array (must have batch*channels*height*width elements)
     */
    Tensor(int batch, int channels, int height, int width, const float* data);
    
    /**
     * Destructor - Releases allocated memory
     */
    ~Tensor();
    
    // Prevent copying by default (implement copy constructor if needed)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // ========================================================================
    // DIMENSION ACCESSORS
    // ========================================================================
    
    /// Get batch size (first dimension)
    int batch() const { return dims_[0]; }
    
    /// Get number of channels (second dimension for 4D, or 1 for 2D)
    int channels() const { return dims_.size() > 1 ? dims_[1] : 1; }
    
    /// Get height (third dimension for 4D, or 1 otherwise)
    int height() const { return dims_.size() > 2 ? dims_[2] : 1; }
    
    /// Get width (fourth dimension for 4D, or 1 otherwise)
    int width() const { return dims_.size() > 3 ? dims_[3] : 1; }
    
    /// Get total number of elements (product of all dimensions)
    int size() const { return total_size_; }
    
    /// Get number of dimensions (2 or 4)
    int ndims() const { return dims_.size(); }
    
    // ========================================================================
    // DATA ACCESS
    // ========================================================================
    
    /// Get raw pointer to data (mutable)
    float* data() { return data_.get(); }
    
    /// Get raw pointer to data (const)
    const float* data() const { return data_.get(); }
    
    // ========================================================================
    // ELEMENT ACCESS (4D indexing)
    // ========================================================================
    
    /**
     * Access 4D tensor element: tensor(batch, channel, height, width)
     * 
     * @return Reference to element
     */
    float& operator()(int b, int c, int h, int w);
    const float& operator()(int b, int c, int h, int w) const;
    
    // ========================================================================
    // ELEMENT ACCESS (2D indexing)
    // ========================================================================
    
    /**
     * Access 2D tensor element: tensor(batch, feature)
     * 
     * @return Reference to element
     */
    float& operator()(int b, int f);
    const float& operator()(int b, int f) const;
    
    // ========================================================================
    // UTILITY OPERATIONS
    // ========================================================================
    
    /**
     * Fill entire tensor with a constant value
     * 
     * @param value - Value to fill with
     */
    void fill(float value);
    
    /**
     * Fill entire tensor with zeros 
     */
    void zeros();
    
private:
    // ========================================================================
    // PRIVATE MEMBERS
    // ========================================================================
    
    std::vector<int> dims_;        // Stores dimensions [batch, channels, height, width]
    int total_size_;               // Cached total number of elements
    std::unique_ptr<float[]> data_; // Underlying data storage

};
