// ============================================================================
// Tensor Implementation
// ============================================================================
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
    total_size_ = batch * channels * height * width;
    data_ = std::make_unique<float[]>(total_size_);
    std::fill(data_.get(), data_.get() + total_size_, 0.0f); // initialize to 0
}

Tensor::Tensor(int batch, int features)
    : dims_({batch, features})
{
    total_size_ = batch * features;
    data_ = std::make_unique<float[]>(total_size_);
    std::fill(data_.get(), data_.get() + total_size_, 0.0f); // initialize to 0
}

Tensor::Tensor(int batch, int channels, int height, int width, const float* data)
    : Tensor(batch, channels, height, width)
{
    std::copy(data, data + total_size_, data_.get());
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
    // Row-major order: index = b * (C*H*W) + c * (H*W) + h * W + w
    int index = b * dims_[1] * dims_[2] * dims_[3] + c * dims_[2] * dims_[3] + h * dims_[3] + w;
    return data_[index];
}

const float& Tensor::operator()(int b, int c, int h, int w) const
{
    int index = b * dims_[1] * dims_[2] * dims_[3] + c * dims_[2] * dims_[3] + h * dims_[3] + w;
    return data_[index];
}

// ============================================================================
// ELEMENT ACCESS (2D)
// ============================================================================

float& Tensor::operator()(int b, int f)
{
    // index = b * F + f
    int index = b * dims_[1] + f;
    return data_[index];
}

const float& Tensor::operator()(int b, int f) const
{
    int index = b * dims_[1] + f;
    return data_[index];
}

// ============================================================================
// UTILITY OPERATIONS
// ============================================================================

void Tensor::fill(float value)
{
    std::fill(data_.get(), data_.get() + total_size_, value);
}

void Tensor::zeros()
{
    fill(0.0f);
}

