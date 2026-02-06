// ============================================================================
// Tensor Implementation
// ============================================================================
// 
// TODO: Implement all methods declared in include/tensor.h
// 
// Key implementation details:
// 1. Use std::unique_ptr for automatic memory management
// 2. Store dimensions in a vector for flexibility
// 3. Use row-major (C-style) memory layout: last dimension changes fastest
// 4. Implement efficient linear indexing in compute_index()
//

#include "tensor.h"
#include <random>
#include <cmath>
#include <algorithm>

// ============================================================================
// CONSTRUCTORS & DESTRUCTORS
// ============================================================================

Tensor::Tensor(int batch, int channels, int height, int width)
    : dims_({batch, channels, height, width})
{
    // TODO: Calculate total_size_ = batch * channels * height * width
    // TODO: Allocate data_ with cudaHostAlloc or std::make_unique<float[]>
}

Tensor::Tensor(int batch, int features)
    : dims_({batch, features})
{
    // TODO: Calculate total_size_ = batch * features
    // TODO: Allocate data_
}

Tensor::Tensor(int batch, int channels, int height, int width, const float* data)
    : Tensor(batch, channels, height, width)
{
    // TODO: Copy data from external pointer using std::copy or memcpy
}

Tensor::~Tensor()
{
    // TODO: Clean up is handled automatically by std::unique_ptr
    // But if using cudaHostAlloc, manually free here
}

// ============================================================================
// ELEMENT ACCESS (4D)
// ============================================================================

float& Tensor::operator()(int b, int c, int h, int w)
{
    // TODO: Calculate linear index and return reference to data_[index]
    // Row-major order: index = b * (C*H*W) + c * (H*W) + h * W + w
    throw std::runtime_error("Not implemented");
}

const float& Tensor::operator()(int b, int c, int h, int w) const
{
    // TODO: Same as mutable version but for const access
    throw std::runtime_error("Not implemented");
}

// ============================================================================
// ELEMENT ACCESS (2D)
// ============================================================================

float& Tensor::operator()(int b, int f)
{
    // TODO: Calculate linear index for 2D tensor
    // index = b * F + f
    throw std::runtime_error("Not implemented");
}

const float& Tensor::operator()(int b, int f) const
{
    // TODO: Same as mutable version but for const access
    throw std::runtime_error("Not implemented");
}

// ============================================================================
// UTILITY OPERATIONS
// ============================================================================

void Tensor::fill(float value)
{
    // TODO: Use std::fill to set all elements to value
}

void Tensor::zeros()
{
    // TODO: Call fill(0.0f)
}

void Tensor::randn(float mean, float std)
{
    // TODO: Generate normally distributed random values
    // Use std::mt19937 and std::normal_distribution
    // Or Box-Muller transform for efficiency
}

void Tensor::relu_inplace()
{
    // TODO: For each element: data_[i] = max(0.0f, data_[i])
}

Tensor Tensor::clone() const
{
    // TODO: Create new tensor with same dimensions
    // Copy all data from this tensor to new tensor
    // Return the new tensor
    throw std::runtime_error("Not implemented");
}

// ============================================================================
// PRIVATE HELPERS
// ============================================================================

int Tensor::compute_index(const std::vector<int>& indices) const
{
    // TODO: Calculate linear index from multi-dimensional indices
    // Use row-major (C-style) ordering:
    // For 4D: index = b * (C*H*W) + c * (H*W) + h * W + w
    // For 2D: index = b * F + f
    throw std::runtime_error("Not implemented");
}
